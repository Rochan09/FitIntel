# Database Persistence Setup for Render

## Problem: User Credentials Not Persisting

**Issue**: Users are lost every time the Render app restarts because SQLite doesn't persist on Render's ephemeral filesystem.

**Solution**: Use PostgreSQL database for persistent user storage.

## 🚀 Quick Setup Steps

### 1. Create PostgreSQL Database on Render

1. Go to your [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"PostgreSQL"**
3. Configure:
   - **Name**: `fitintel-database`
   - **Database**: `fitintel`
   - **User**: `fitintel_user`
   - **Region**: Same as your web service
   - **Plan**: **Free Tier** ($0/month)

4. Click **"Create Database"**
5. Wait for database to be created (2-3 minutes)

### 2. Get Database Connection String

1. Go to your PostgreSQL database in Render dashboard
2. Click on **"Connect"** tab
3. Copy the **"External Database URL"**
   - Format: `postgresql://user:password@host:port/database`

### 3. Configure Web Service Environment Variable

1. Go to your **FitIntel web service** in Render dashboard
2. Go to **"Environment"** tab
3. Add new environment variable:
   - **Key**: `DATABASE_URL`
   - **Value**: Paste the PostgreSQL URL from step 2
4. Click **"Save Changes"**

### 4. Redeploy Your App

1. The app will automatically redeploy with the new database
2. Check logs to see: `🐘 Using PostgreSQL database for persistent user storage`
3. Visit `/health` endpoint to verify: `"persistent": true`

## 🎯 Verification Steps

### Test User Persistence:

1. **Register** a new user on your app
2. **Log in** successfully  
3. **Force restart** the app (go to Render dashboard → redeploy)
4. **Try logging in** with same credentials
5. **✅ Success**: User data should persist!

### Health Check:

Visit: `https://your-app.onrender.com/health`

Should show:
```json
{
  "database": {
    "connected": true,
    "type": "PostgreSQL", 
    "persistent": true,
    "note": "User data persists across deployments ✅"
  }
}
```

## 🆓 Cost: **FREE**

- **PostgreSQL Free Tier**: $0/month
- **Limitations**: 1GB storage, 100 connections
- **Perfect for**: User authentication and basic data

## 🔧 Troubleshooting

### If you see SQLite warnings:
- Environment variable `DATABASE_URL` not set correctly
- Check Render dashboard → Environment tab
- Ensure PostgreSQL database is running

### If database connection fails:
- Wait 2-3 minutes for PostgreSQL to fully initialize
- Check PostgreSQL database status in Render dashboard
- Verify connection string format

## 📊 Before vs After

| Aspect | Before (SQLite) | After (PostgreSQL) |
|--------|----------------|-------------------|
| User persistence | ❌ Lost on restart | ✅ Permanent |
| Database location | Ephemeral filesystem | Persistent cloud DB |
| Setup complexity | Simple | One-time setup |
| Cost | Free | Free |
| Reliability | Poor | Excellent |

## 🎉 Result

**User credentials will now persist permanently** across all app restarts, redeployments, and updates!