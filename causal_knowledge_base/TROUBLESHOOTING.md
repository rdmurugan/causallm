# 🔧 Troubleshooting Guide - CausalLLM Streamlit App

## Issue: Empty Pages (except Home)

### Symptoms
- Home page loads correctly
- Other pages appear empty
- "Deploy" button visible in top-right corner

### Root Causes & Solutions

#### 1. **Import Errors (Most Common)**

**Diagnosis:**
```bash
# Run the startup check script
python startup_check.py
```

**Common Import Issues:**
- Missing dependencies: `pip install streamlit pandas numpy plotly networkx scipy`
- CausalLLM library import failures
- Path issues with the CausalLLM library

**Solution:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify CausalLLM library is accessible
python -c "import causalllm.core; print('CausalLLM OK')"

# If CausalLLM import fails, make sure you're in the right directory
cd /Users/durai/Documents/GitHub/causallm
python -c "import causalllm.core; print('CausalLLM OK')"
```

#### 2. **Path Issues**

**Diagnosis:**
The app may not be finding the CausalLLM library correctly.

**Solution:**
```bash
# Always run from the project root directory
cd /Users/durai/Documents/GitHub/causallm
cd streamlit_app
streamlit run main.py
```

#### 3. **Enhanced Error Reporting**

With the updated `main.py`, pages that fail to load will now show detailed error messages instead of appearing empty.

**What you'll see now:**
- ✅ Working pages: Full functionality
- ❌ Failed pages: Error message with troubleshooting steps
- 🧪 Test page: Always works (for debugging)

### Quick Fix Steps

1. **Run Startup Check:**
   ```bash
   python startup_check.py
   ```

2. **Start App with Enhanced Error Handling:**
   ```bash
   cd streamlit_app
   streamlit run main.py
   ```

3. **Check Error Messages:**
   - Navigate to any page that was previously empty
   - You should now see detailed error information
   - Follow the troubleshooting steps shown

### Debugging Mode

If pages are still empty, try the **Test Page** first:
1. Select "🧪 Test Page" from the navigation
2. If this works, the app infrastructure is fine
3. Other pages have import/dependency issues

### Common Solutions

#### Missing Dependencies
```bash
pip install streamlit pandas numpy plotly networkx scipy asyncio
```

#### CausalLLM Library Issues
```bash
# Make sure you're in the right directory
cd /Users/durai/Documents/GitHub/causallm

# Test the library
python -c "from causalllm.core import CausalLLMCore; print('Success!')"
```

#### Environment Issues
```bash
# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Ensure the project directory is in path
export PYTHONPATH="/Users/durai/Documents/GitHub/causallm:$PYTHONPATH"
```

### Advanced Debugging

#### Check Individual Page Imports
```bash
cd streamlit_app
python -c "
import sys
sys.path.append('..')
from pages import data_manager
print('data_manager import successful')
"
```

#### Run Pages Directly
```bash
# Test if page modules have show() function
python -c "
import sys
sys.path.append('..')
from pages import home
if hasattr(home, 'show'):
    print('home.show() exists')
else:
    print('home.show() missing')
"
```

### Expected Behavior After Fix

**Before Fix:**
- Home: ✅ Working
- Other pages: Empty (white page)

**After Fix:**
- Home: ✅ Working  
- Working pages: ✅ Full functionality
- Failed pages: ❌ Detailed error message with solutions
- Test page: ✅ Always works for debugging

### Still Having Issues?

1. **Check the browser console** for JavaScript errors
2. **Try a different browser** or incognito mode
3. **Clear browser cache** (Ctrl+F5 or Cmd+Shift+R)
4. **Restart the Streamlit server**:
   ```bash
   # Kill any existing processes
   pkill -f streamlit
   
   # Start fresh
   cd streamlit_app
   streamlit run main.py
   ```

5. **Check Streamlit logs** in the terminal for error messages

### Contact Support

If you're still experiencing issues after trying these solutions:

1. Run the startup check: `python startup_check.py`
2. Share the output of the startup check
3. Include any error messages from the Streamlit app
4. Specify which pages are working vs. failing

---

**Last Updated**: August 2024  
**Version**: CausalLLM Pro v2.0