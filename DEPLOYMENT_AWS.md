# AWS EC2 Deployment Guide: Docker + GitHub Actions

This guide explains how to deploy this MLOps project to an AWS EC2 instance using Docker Compose and GitHub Actions for Continuous Deployment (CD).

## 🏗️ Architecture Overview
- **GitHub Actions**: Triggers on push to `main`, builds images, and SSHs into EC2 to deploy.
- **AWS EC2**: Hosts the entire stack (API, Frontend, Airflow, Postgres) inside Docker containers.
- **Docker Compose**: Orchestrates the multi-container environment.
- **DVC/S3**: Handles data and model versioning.

---

## 1. ☁️ AWS Setup (One-Time)

### A. Launch EC2 Instance
- **Instance Type**: `t3.medium` (minimum 4GB RAM recommended for Airflow).
- **OS**: Ubuntu 22.04 LTS.
- **Security Group**: Open these ports:
  - `22` (SSH)
  - `80` (HTTP) / `443` (HTTPS)
  - `8000` (FastAPI)
  - `8501` (Streamlit)
  - `8080` (Airflow UI)

### B. Prepare the EC2 Environment
SSH into your instance and run:
```bash
# Update and install Docker
sudo apt-get update
sudo apt install docker.io -y
sudo apt install docker-compose -y
sudo usermod -aG docker $USER
newgrp docker
```

### C. Create IAM Role
Attach a role to the EC2 instance with `AmazonS3FullAccess` so DVC can pull models/data from your S3 bucket.

---

## 2. 🔐 GitHub Secrets
In your GitHub Repo, go to **Settings > Secrets and variables > Actions** and add:
- `EC2_SSH_KEY`: Your `.pem` private key.
- `HOST_DNS`: The Public DNS of your EC2 (e.g., `ec2-xx-xx-xx.compute.amazonaws.com`).
- `USERNAME`: `ubuntu`
- `DATABASE_URL`: `postgresql://postgres:postgres@postgres:5432/churn`

---

## 3. 🚀 Deployment Commands (Manual)
If you want to deploy manually from the EC2 terminal:
```bash
# Clone the repo (first time)
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Pull DVC artifacts (Requires AWS CLI/Role)
dvc pull

# Start all services
docker-compose up -d --build
```

---

## 4. 🔄 CI/CD Workflow (`cd.yml`)
The project uses a GitHub Action to automate this. Every time you push to `main`:
1. It runs tests.
2. It SSHs into EC2.
3. It pulls the latest code.
4. It restarts the containers.

---

## 📝 Critical Notes for Future Projects
1. **Docker Compose is King**: Always use `docker-compose` for local and remote consistency.
2. **Health Checks**: Ensure your DB is ready before the API starts (use `depends_on` + healthcheck).
3. **Persist Data**: Always use Docker Volumes for Postgres (`/var/lib/postgresql/data`) or you will lose your data on restart.
4. **Environment Variables**: Never hardcode credentials. Use `.env` files or GitHub Secrets.
5. **Reverse Proxy**: In a real production app, put **Nginx** in front of your services to handle SSL (HTTPS).

## 🛠️ Common Troubleshooting
- **Out of Memory**: If `docker-compose build` fails, your EC2 might need a swap file or a larger instance.
- **Connection Refused**: Check your AWS Security Group ports.
- **DVC Pull Failure**: Ensure the IAM Role is correctly attached to the EC2 instance.
