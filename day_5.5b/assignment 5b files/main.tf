terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 3.20.0"
    }
  }
}

provider "aws" {
  profile = var.aws_tf_profile
  region  = var.region
}

module "s3_module" {
  source      = "./s3_module"
  bucket_name = var.bucket_name
}

resource "aws_s3_bucket_object" "s3_object" {
  bucket = module.s3_module.s3_bucket_name
  key    = "day2/IaC/"
}


output "s3-bucket-id" {
  value = module.s3_module.s3_bucket_name
} 