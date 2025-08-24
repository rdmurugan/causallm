# 🚀 Quick Start Guide - CausalLLM Streamlit App

## ✅ **Current Status: Fixed and Working**

Your CausalLLM Streamlit app is now properly configured and should be working correctly.

## 🎯 **How to Run the App**

### **Method 1: Fixed Version (Recommended)**
```bash
cd /Users/durai/Documents/GitHub/causallm/streamlit_app
streamlit run main_fixed.py
```

### **Method 2: Replace Main File**
```bash
cd /Users/durai/Documents/GitHub/causallm/streamlit_app
cp main_fixed.py main.py  # Replace the original
streamlit run main.py
```

## 🌐 **Access the App**

Once running, the app will be available at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.86.221:8501 (accessible from other devices on your network)

## 📱 **What You Should See**

### **✅ Expected Behavior:**
1. **🏠 Home Page**: Loads first with welcome content and sample datasets
2. **📊 Navigation**: All pages accessible via sidebar dropdown
3. **🔧 Debug Mode**: Optional debug info (toggle in sidebar)
4. **📍 Page Descriptions**: Each page shows what it does

### **🎮 Navigation:**
- **Sidebar**: Select pages from dropdown menu
- **Home**: Welcome, features overview, sample data
- **Data Manager**: Upload and analyze datasets
- **Causal Discovery**: AI-powered relationship discovery
- **Interactive Q&A**: Natural language queries
- **And 6 more advanced pages...**

## 🔧 **About the "Deploy" Button**

The "Deploy" button you see in the top-right corner is for **Streamlit Community Cloud** (online hosting). You can:

### **Ignore It (Recommended for Now)**
- It's not needed to run the app locally
- The app works perfectly without it
- Focus on using the features first

### **Or Deploy Later (Optional)**
If you want to share your app online:

1. **Create GitHub Repository:**
   ```bash
   cd /Users/durai/Documents/GitHub/causallm
   git init
   git add .
   git commit -m "CausalLLM Streamlit App"
   ```

2. **Push to GitHub:**
   ```bash
   # Create repository on GitHub first, then:
   git remote add origin https://github.com/yourusername/causallm.git
   git push -u origin main
   ```

3. **Use Deploy Button:**
   - Click "Deploy" in the app
   - Connect to your GitHub repository
   - Streamlit will host it for free

## 🎯 **Getting Started with the App**

### **First Steps:**
1. **🏠 Start with Home** - Overview and sample data
2. **📊 Upload Data** - Use Data Manager to load your dataset
3. **🔍 Discover Relationships** - Try Causal Discovery with AI
4. **💬 Ask Questions** - Use Interactive Q&A in natural language

### **Sample Workflow:**
```
Home → Load Sample Data → Data Manager → Assign Variables → 
Causal Discovery → View Results → Interactive Q&A → Ask Questions
```

## 🚨 **Troubleshooting**

### **If Pages Are Empty:**
1. **Check Debug Mode** - Enable in sidebar to see module status
2. **Look for Error Messages** - Pages now show detailed errors instead of being blank
3. **Try Different Pages** - Some might work while others show errors

### **If App Won't Start:**
```bash
# Check dependencies
python startup_check.py

# Kill existing processes
pkill -f streamlit

# Start fresh
streamlit run main_fixed.py
```

### **If Import Errors:**
```bash
# Ensure you're in the right directory
cd /Users/durai/Documents/GitHub/causallm/streamlit_app

# Check CausalLLM library
python -c "import causalllm.core; print('✅ CausalLLM works')"
```

## 🎉 **Success Indicators**

You'll know it's working when you see:
- ✅ **Home page loads** with welcome content
- ✅ **All pages accessible** via sidebar navigation  
- ✅ **No empty pages** - either content or clear error messages
- ✅ **Debug info available** when enabled

## 📞 **Need Help?**

1. **Enable Debug Mode** in the sidebar
2. **Check Module Status** in the debug section
3. **Look at Error Messages** - they now provide specific guidance
4. **Try the Simple Test**: `streamlit run simple_test.py`

---

**Your app should now be fully functional! 🎉**

Open http://localhost:8501 in your browser and explore the CausalLLM features through the beautiful UI interface.