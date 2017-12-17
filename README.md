Object Tracking
====================

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Make sure the video(demo.avi) and background(demo_0.png) files are there
```
87568 -rw-r--r-- 1 User 197609 89666412 demo.avi
 1808 -rw-r--r-- 1 User 197609  1849409 demo_0.png 
```

3. Run the script
```bash
python tracking.py demo.avi
```

4. Key Parameters
In tracking.py,
line 7: interested area
line 8: alpha correction coefficient