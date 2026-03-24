# Deployment Guide

This guide provides comprehensive instructions for deploying the Prioritization Frameworks Web Application to production environments.

## Quick Deploy

### Frontend (Vercel)

1. Install Vercel CLI: `npm i -g vercel`
2. Login: `vercel login`
3. Deploy: `cd frontend && vercel --prod`

### Backend (Railway)

1. Install Railway CLI: `npm i -g @railway/cli`
2. Login: `railway login`
3. Deploy: `cd backend && railway up`

## Environment Variables

Set the following in your deployment platform:

### Frontend (Vercel)
- `VITE_API_URL` - Backend API URL (e.g., `https://api.yourdomain.com/api`)
- `VITE_WS_URL` - WebSocket URL (e.g., `wss://api.yourdomain.com/ws`)

### Backend (Railway)
- `PORT` - Server port (default: 3001)
- `CORS_ORIGIN` - Frontend URL for CORS (e.g., `https://yourdomain.com`)
- `NODE_ENV` - Set to `production`

## Docker Deployment

### Local Development

```bash
# Build and run with Docker Compose
cd src/prioritization-app
docker-compose up --build
```

This will start:
- Frontend on http://localhost:5173
- Backend on http://localhost:3001

### Production Deployment

```bash
# Create production environment files
cp frontend/.env.production.example frontend/.env.production
cp backend/.env.production.example backend/.env.production

# Edit environment files with production values

# Deploy with Docker Compose
docker-compose -f docker-compose.yml up -d
```

### Docker Commands

```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# Run specific service
docker-compose up backend
```

## CI/CD Pipeline

GitHub Actions automatically:

1. **Runs linting and tests** on every push to `main` or `develop` branches
2. **Builds frontend and backend** on PR merge to `main`
3. **Deploys to Vercel** (frontend) and **Railway** (backend) on main branch
4. **Runs E2E tests** against production deployment

### Required GitHub Secrets

Set these secrets in your GitHub repository settings:

| Secret | Description |
|--------|-------------|
| `VERCEL_TOKEN` | Vercel authentication token |
| `VERCEL_ORG_ID` | Vercel organization ID |
| `VERCEL_PROJECT_ID` | Vercel project ID |
| `RAILWAY_TOKEN` | Railway authentication token |
| `RAILWAY_PROJECT_ID` | Railway project ID |

### Pipeline Stages

```
push -> lint-and-test -> build -> deploy-frontend -> e2e-tests
                              -> deploy-backend -^
```

## Manual Build Commands

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Analyze bundle size
npm run build:analyze

# Preview production build
npm run preview
```

### Backend

```bash
cd backend

# Install dependencies
npm install

# Run development server
npm run dev

# Build TypeScript
npm run build

# Run production server
npm run start
```

## Health Checks

Monitor your deployment with these endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Backend health check |
| `GET /api/v1/stats` | System statistics |
| `GET /` | Frontend availability |

### Example Health Check

```bash
# Check backend health
curl https://api.yourdomain.com/api/health

# Expected response
{"status":"ok","timestamp":"2024-01-01T00:00:00.000Z"}
```

## Monitoring

### Vercel Dashboard
- View deployment logs at https://vercel.com/dashboard
- Monitor performance and analytics
- Configure alerts and notifications

### Railway Dashboard
- View deployment logs at https://railway.app/dashboard
- Monitor resource usage (CPU, memory)
- Configure environment variables

### Application Logs

```bash
# Docker logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Set log level via environment
# backend: LOG_LEVEL=debug|info|warn|error
```

## Troubleshooting

### Frontend Issues

**Blank page after deployment:**
1. Check browser console for errors
2. Verify `VITE_API_URL` is correct
3. Check Vercel build logs for errors

**API calls failing:**
1. Verify backend is deployed and running
2. Check CORS settings in backend
3. Ensure `VITE_API_URL` matches backend URL

### Backend Issues

**Server not starting:**
1. Check Railway deployment logs
2. Verify `PORT` environment variable
3. Ensure `npm run build` completes successfully

**CORS errors:**
1. Set `CORS_ORIGIN` to frontend URL
2. For multiple origins, update CORS configuration in `backend/src/index.ts`

### Docker Issues

**Container fails to start:**
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs <service-name>

# Rebuild containers
docker-compose up --build --force-recreate
```

**Port already in use:**
```bash
# Change ports in docker-compose.yml
ports:
  - "5174:80"  # Frontend
  - "3002:3001" # Backend
```

## Scaling Considerations

### Frontend (Vercel)
- Automatic scaling with Vercel's edge network
- CDN caching for static assets
- Consider enabling Incremental Static Regeneration (ISR) for dynamic content

### Backend (Railway)
- Enable auto-scaling in Railway dashboard
- Consider adding Redis for session management
- Use database connection pooling

### Database
- Use Railway's managed PostgreSQL for production
- Enable automated backups
- Monitor connection limits

## Security Best Practices

1. **Environment Variables**: Never commit `.env` files
2. **API Keys**: Use GitHub secrets for sensitive data
3. **HTTPS**: Enforced by default on Vercel and Railway
4. **CORS**: Restrict to known frontend domains
5. **Rate Limiting**: Implement in backend for API protection

## Support

For issues or questions:
1. Check deployment logs in Vercel/Railway dashboards
2. Review GitHub Actions workflow runs
3. Consult the application README.md
