# Docker Deployment Guide

Detta dokument beskriver hur du kör Fotbollspredictor v7 med Docker.

## Snabbstart

### Förutsättningar
- Docker installerat (version 20.10+)
- Docker Compose installerat (version 2.0+)

### Steg 1: Klona repositoryt
```bash
git clone https://github.com/Emtatos/fotbollspredictor_v7.git
cd fotbollspredictor_v7
```

### Steg 2: Konfigurera miljövariabler
Skapa en `.env`-fil i projektets rot:
```bash
cp .env.example .env
# Redigera .env och lägg till dina API-nycklar
```

### Steg 3: Bygg och starta containern
```bash
docker-compose up -d
```

Applikationen är nu tillgänglig på `http://localhost:8501`

## Användning

### Starta applikationen
```bash
docker-compose up -d
```

### Stoppa applikationen
```bash
docker-compose down
```

### Visa loggar
```bash
docker-compose logs -f app
```

### Återbygga efter kodändringar
```bash
docker-compose up -d --build
```

### Köra kommandon i containern
```bash
# Träna om modellen
docker-compose exec app python main.py

# Köra tester
docker-compose exec app pytest tests/ -v

# Öppna Python-shell
docker-compose exec app python
```

## Produktion

### Bygg produktionsbild
```bash
docker build -t fotbollspredictor:latest .
```

### Kör i produktion
```bash
docker run -d \
  --name fotbollspredictor \
  -p 8501:8501 \
  -e API_FOOTBALL_KEY="your_key" \
  -e OPENAI_API_KEY="your_key" \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --restart unless-stopped \
  fotbollspredictor:latest
```

### Deployment till molntjänster

#### Docker Hub
```bash
# Logga in
docker login

# Tagga bilden
docker tag fotbollspredictor:latest yourusername/fotbollspredictor:latest

# Pusha till Docker Hub
docker push yourusername/fotbollspredictor:latest
```

#### Render.com (med Docker)
1. Gå till Render Dashboard
2. Välj "New Web Service"
3. Anslut ditt GitHub-repo
4. Välj "Docker" som runtime
5. Lägg till miljövariabler
6. Klicka "Create Web Service"

#### AWS ECS/Fargate
```bash
# Installera AWS CLI och konfigurera
aws configure

# Skapa ECR-repository
aws ecr create-repository --repository-name fotbollspredictor

# Logga in på ECR
aws ecr get-login-password --region eu-north-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.eu-north-1.amazonaws.com

# Tagga och pusha
docker tag fotbollspredictor:latest YOUR_ACCOUNT_ID.dkr.ecr.eu-north-1.amazonaws.com/fotbollspredictor:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.eu-north-1.amazonaws.com/fotbollspredictor:latest
```

## Felsökning

### Container startar inte
```bash
# Kontrollera loggar
docker-compose logs app

# Verifiera att alla filer finns
docker-compose exec app ls -la
```

### Port redan i bruk
Om port 8501 redan används, ändra i `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"  # Använd port 8502 istället
```

### Modellen tränas inte
```bash
# Träna manuellt
docker-compose exec app python main.py
```

### API-nycklar fungerar inte
```bash
# Verifiera miljövariabler
docker-compose exec app env | grep API
```

## Optimering

### Minska bildstorlek
Bilden använder redan multi-stage build, men du kan optimera ytterligare:
- Ta bort onödiga dependencies från `requirements.txt`
- Använd Alpine Linux istället för Debian (kräver kompilering av vissa paket)

### Förbättra prestanda
- Använd Docker BuildKit: `DOCKER_BUILDKIT=1 docker build .`
- Cacha Python-paket: Lägg till `--mount=type=cache` i Dockerfile
- Använd mindre base image: `python:3.11-slim-bullseye`

### Säkerhet
- Kör inte som root: Lägg till `USER` i Dockerfile
- Scanna för sårbarheter: `docker scan fotbollspredictor:latest`
- Använd secrets istället för miljövariabler i produktion

## Resurser

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Docker Deployment](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Best Practices för Python Docker Images](https://docs.docker.com/language/python/build-images/)
