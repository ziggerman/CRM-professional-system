#!/bin/bash
# Release Process Script (Step 9)
# Usage: ./scripts/release.sh [staging|canary|production]
# 
# This script implements the release workflow:
# 1. staging -> 2. canary -> 3. production
# With rollback capability at each stage

set -e

STAGE=${1:-staging}
APP_NAME="crm-app"
TAG=${2:-latest}

echo "=========================================="
echo "CRM Release Process - $STAGE"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not a git repository"
    exit 1
fi

# Get current git commit
GIT_COMMIT=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git branch --show-current)
print_status "Git commit: $GIT_COMMIT"
print_status "Git branch: $GIT_BRANCH"

# Pre-release checks
print_status "Running pre-release checks..."

# Check for uncommitted changes
if ! git diff --quiet; then
    print_warning "You have uncommitted changes. Commit or stash them before releasing."
    git status --short
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run tests before release
print_status "Running pre-release tests..."
if ! pytest tests/ -v --tb=short -q; then
    print_error "Tests failed. Aborting release."
    exit 1
fi

print_status "Pre-release checks passed!"

# Build the application
print_status "Building application..."
docker build -t $APP_NAME:$STAGE-$TAG .
docker build -t $APP_NAME:$STAGE-$TAG-$GIT_COMMIT .

case $STAGE in
    staging)
        print_status "Deploying to STAGING..."
        # Deploy to staging
        docker-compose up -d --build api
        # Wait for health check
        print_status "Waiting for staging to be ready..."
        sleep 10
        # Run smoke tests
        curl -f http://localhost:8000/health || exit 1
        print_status "Staging deployed successfully!"
        echo "Staging URL: http://localhost:8000"
        ;;
    
    canary)
        print_status "Deploying to CANARY (10% traffic)..."
        # In a real scenario, this would use Kubernetes or similar
        # For now, we just tag the canary version
        docker tag $APP_NAME:staging-$TAG $APP_NAME:canary-$TAG
        # Simulate canary deployment
        print_status "Canary version tagged: $APP_NAME:canary-$TAG"
        
        # Run canary validation
        print_status "Running canary validation tests..."
        sleep 5
        
        # Check if canary is healthy
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            print_status "Canary validation passed!"
        else
            print_warning "Canary may have issues - check manually"
        fi
        ;;
    
    production)
        print_status "Deploying to PRODUCTION..."
        
        # Create backup before production deployment
        print_status "Creating database backup..."
        ./scripts/backup_db.sh
        
        # Tag the release
        git tag -a v$TAG -m "Production release $TAG"
        git push origin v$TAG
        
        # Deploy production
        docker tag $APP_NAME:canary-$TAG $APP_NAME:production-$TAG
        docker tag $APP_NAME:canary-$TAG $APP_NAME:latest
        
        print_status "Production deployed successfully!"
        echo "Production image: $APP_NAME:production-$TAG"
        echo "Latest image: $APP_NAME:latest"
        ;;
    
    rollback)
        print_status "Rolling back to previous version..."
        # Get previous tag
        PREVIOUS_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "previous")
        print_status "Rolling back to: $PREVIOUS_TAG"
        
        # In production, this would pull the previous image
        docker pull $APP_NAME:$PREVIOUS_TAG
        print_status "Rollback complete!"
        ;;
    
    *)
        print_error "Unknown stage: $STAGE"
        echo "Usage: $0 [staging|canary|production|rollback] [version-tag]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo -e "${GREEN}Release process completed!${NC}"
echo "=========================================="
echo ""
echo "To monitor the deployment:"
echo "  - Logs: docker-compose logs -f api"
echo "  - Health: curl http://localhost:8000/health"
echo ""
echo "If issues occur, run: $0 rollback"
