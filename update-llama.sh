#!/bin/bash
# Update models, pull latest images, restart llama.cpp + open-webui stack
set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

trap 'echo -e "${RED}ERROR on line $LINENO${NC}"' ERR

echo "=== llama.cpp Stack Update Script ==="
echo ""

COMPOSE_FILE="$HOME/docker/llama-openwebui/docker-compose.yml"
MODELS_DIR="$HOME/models"
DOWNLOAD_DIR="$HOME/hf-model-download"
SKIP_MODELS=false

for arg in "$@"; do
    case "$arg" in
        --skip-models) SKIP_MODELS=true ;;
        *) echo -e "${RED}Unknown option: $arg${NC}"; echo "Usage: $0 [--skip-models]"; exit 1 ;;
    esac
done

require() { command -v "$1" >/dev/null 2>&1 || { echo -e "${RED}Missing: $1${NC}"; exit 1; }; }
for cmd in docker curl python3; do
    require "$cmd"
done

[ -f "$COMPOSE_FILE" ]         || { echo -e "${RED}ERROR: Compose file not found: $COMPOSE_FILE${NC}"; exit 1; }
[ -f "$DOWNLOAD_DIR/run.sh" ]  || { echo -e "${RED}ERROR: run.sh not found in $DOWNLOAD_DIR${NC}"; exit 1; }

# Step 1: Update models
if [ "$SKIP_MODELS" = true ]; then
    echo -e "${YELLOW}Step 1: Skipping model update (--skip-models)${NC}"
else
    echo "Step 1: Checking and updating models..."
    mkdir -p "$MODELS_DIR"
    if ! "$DOWNLOAD_DIR/run.sh" --output-dir "$MODELS_DIR"; then
        echo -e "${RED}ERROR: Model download reported failures${NC}"
        echo -e "${RED}Aborting container update; rerun with --skip-models to update images only.${NC}"
        exit 1
    else
        echo -e "${GREEN}✓ Models up to date${NC}"
    fi
fi

# Step 2: Pull latest images
echo ""
echo "Step 2: Pulling latest container images..."
docker compose -f "$COMPOSE_FILE" pull
echo -e "${GREEN}✓ Images pulled${NC}"

# Step 3: Recreate containers (only those whose images changed)
echo ""
echo "Step 3: Recreating containers..."
docker compose -f "$COMPOSE_FILE" up -d
echo -e "${GREEN}✓ Services started${NC}"

# Step 4: Wait for health
echo ""
echo "Step 4: Waiting for services to become healthy..."
SERVICES=("llama-tasks" "llama-qwen36-thinking" "open-webui" "open-terminal")
MAX_WAIT=240
for svc in "${SERVICES[@]}"; do
    echo -n "  $svc: "
    iterations=$((MAX_WAIT / 2))
    healthy=false
    for i in $(seq 1 $iterations); do
        status=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$svc" 2>/dev/null || echo "missing")
        case "$status" in
            healthy)   echo -e "${GREEN}healthy${NC}"; healthy=true; break ;;
            running)   echo -e "${GREEN}running (no healthcheck)${NC}"; healthy=true; break ;;
            unhealthy) echo -e "${RED}unhealthy${NC}"; docker logs --tail 30 "$svc"; exit 1 ;;
            missing)   echo -e "${RED}missing${NC}"; exit 1 ;;
            exited|dead) echo -e "${RED}${status}${NC}"; docker logs --tail 30 "$svc"; exit 1 ;;
        esac
        sleep 2
    done
    if [ "$healthy" = false ]; then
        echo -e "${RED}timeout after ${MAX_WAIT}s${NC}"
        docker logs --tail 30 "$svc"
        exit 1
    fi
done

# Step 5: Cleanup
echo ""
echo "Step 5: Cleaning up dangling images..."
docker image prune -f
echo -e "${GREEN}✓ Cleanup complete${NC}"

echo ""
echo -e "${GREEN}=== Update Complete ===${NC}"
echo ""
echo "Endpoints:"
echo "  open-webui:            http://localhost:11080"
echo "  open-terminal:         http://localhost:8000"
echo "  llama-qwen36-thinking: http://localhost:8080"
echo "  llama-tasks:           http://localhost:9000"
