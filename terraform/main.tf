# FL Platform — HIPAA-Eligible AWS GovCloud Infrastructure
# Phase 3 — C1 (Clinical-grade cloud infrastructure)
#
# Resources provisioned:
#   - VPC with private/public subnets (no public internet for ECS tasks)
#   - ECS Fargate cluster for fl-server and fl-worker
#   - RDS PostgreSQL (encrypted at rest, in private subnet, Multi-AZ)
#   - ElastiCache Redis (encrypted in transit + at rest)
#   - KMS Customer Managed Key (CMK) for all encryption
#   - ALB with HTTPS (ACM TLS cert)
#   - S3 bucket (encrypted, versioned) for model checkpoints
#   - CloudWatch log groups + metric alarms
#   - IAM roles with least-privilege policies
#
# Compliance notes:
#   - All data-at-rest encryption uses AWS KMS CMK (BAA-eligible)
#   - All network traffic encrypted in transit (TLS 1.2+)
#   - RDS automated backups retained for 30 days
#   - CloudTrail enabled separately at the account level (HIPAA requirement)
#   - VPC Flow Logs enabled for network audit trail
#
# Usage:
#   terraform init
#   terraform plan -var-file=terraform.tfvars
#   terraform apply -var-file=terraform.tfvars

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Remote state (recommended for production)
  # backend "s3" {
  #   bucket         = "fl-platform-terraform-state"
  #   key            = "prod/terraform.tfstate"
  #   region         = var.aws_region
  #   encrypt        = true
  #   dynamodb_table = "fl-platform-terraform-locks"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "fl-platform"
      Environment = var.environment
      Compliance  = "HIPAA"
      ManagedBy   = "Terraform"
    }
  }
}

# ─── KMS Customer Managed Key ──────────────────────────────────────────────────
resource "aws_kms_key" "fl_cmk" {
  description             = "FL Platform CMK — encrypts RDS, S3, ECS secrets"
  deletion_window_in_days = 30
  enable_key_rotation     = true   # HIPAA: annual key rotation

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action   = "kms:*"
        Resource = "*"
      }
    ]
  })
}

resource "aws_kms_alias" "fl_cmk_alias" {
  name          = "alias/fl-platform-${var.environment}"
  target_key_id = aws_kms_key.fl_cmk.key_id
}

# ─── VPC ──────────────────────────────────────────────────────────────────────
module "vpc" {
  source = "./modules/vpc"

  vpc_cidr             = var.vpc_cidr
  availability_zones   = var.availability_zones
  environment          = var.environment
}

# ─── S3 Model Storage ──────────────────────────────────────────────────────────
resource "aws_s3_bucket" "models" {
  bucket        = "fl-platform-models-${var.environment}-${data.aws_caller_identity.current.account_id}"
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.fl_cmk.arn
    }
  }
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ─── RDS PostgreSQL ────────────────────────────────────────────────────────────
resource "aws_db_subnet_group" "fl" {
  name       = "fl-platform-${var.environment}"
  subnet_ids = module.vpc.private_subnet_ids
}

resource "aws_db_instance" "fl_postgres" {
  identifier             = "fl-platform-${var.environment}"
  engine                 = "postgres"
  engine_version         = "16.3"
  instance_class         = var.db_instance_class
  allocated_storage      = 100
  max_allocated_storage  = 1000
  storage_type           = "gp3"
  storage_encrypted      = true
  kms_key_id             = aws_kms_key.fl_cmk.arn
  db_name                = "fl_platform"
  username               = "fl_admin"
  password               = var.db_password   # store in AWS Secrets Manager in prod
  multi_az               = true              # HIPAA: high availability
  backup_retention_period = 30              # HIPAA: 30-day backup retention
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:00-sun:05:00"
  deletion_protection    = true
  skip_final_snapshot    = false
  final_snapshot_identifier = "fl-platform-${var.environment}-final"
  db_subnet_group_name   = aws_db_subnet_group.fl.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  parameter_group_name = aws_db_parameter_group.fl.name
}

resource "aws_db_parameter_group" "fl" {
  name   = "fl-platform-${var.environment}-pg16"
  family = "postgres16"
  parameter {
    name  = "log_connections"
    value = "1"   # HIPAA: audit logging
  }
  parameter {
    name  = "log_disconnections"
    value = "1"
  }
}

# ─── ElastiCache Redis ─────────────────────────────────────────────────────────
resource "aws_elasticache_subnet_group" "fl" {
  name       = "fl-platform-${var.environment}"
  subnet_ids = module.vpc.private_subnet_ids
}

resource "aws_elasticache_replication_group" "fl_redis" {
  replication_group_id = "fl-platform-${var.environment}"
  description          = "FL Platform Redis — pending updates + velocity state"
  node_type            = var.redis_node_type
  num_cache_clusters   = 2           # primary + replica
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.fl.name
  security_group_ids   = [aws_security_group.redis.id]
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  kms_key_id                  = aws_kms_key.fl_cmk.arn
  automatic_failover_enabled  = true
  auth_token                  = var.redis_auth_token
}

# ─── ECS Fargate ───────────────────────────────────────────────────────────────
module "ecs" {
  source = "./modules/ecs"

  environment        = var.environment
  vpc_id             = module.vpc.vpc_id
  private_subnet_ids = module.vpc.private_subnet_ids
  public_subnet_ids  = module.vpc.public_subnet_ids
  kms_key_arn        = aws_kms_key.fl_cmk.arn
  s3_bucket_arn      = aws_s3_bucket.models.arn

  database_url    = "postgresql+psycopg2://fl_admin:${var.db_password}@${aws_db_instance.fl_postgres.endpoint}/fl_platform"
  redis_url       = "rediss://:${var.redis_auth_token}@${aws_elasticache_replication_group.fl_redis.primary_endpoint_address}:6379/0"
  jwt_secret      = var.jwt_secret
  fl_enc_key      = var.fl_encryption_key

  server_image    = var.server_image
  worker_image    = var.worker_image
}

# ─── Security Groups ───────────────────────────────────────────────────────────
resource "aws_security_group" "rds" {
  name   = "fl-platform-rds-${var.environment}"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.ecs.ecs_sg_id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "redis" {
  name   = "fl-platform-redis-${var.environment}"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.ecs.ecs_sg_id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ─── CloudWatch Alarms (HIPAA: operational monitoring) ────────────────────────
resource "aws_cloudwatch_metric_alarm" "rds_cpu" {
  alarm_name          = "fl-rds-high-cpu-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  dimensions = { DBInstanceIdentifier = aws_db_instance.fl_postgres.identifier }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_sns_topic" "alerts" {
  name              = "fl-platform-alerts-${var.environment}"
  kms_master_key_id = aws_kms_key.fl_cmk.arn
}

# ─── Data sources ─────────────────────────────────────────────────────────────
data "aws_caller_identity" "current" {}
