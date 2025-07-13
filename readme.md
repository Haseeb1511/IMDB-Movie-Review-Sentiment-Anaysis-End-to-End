```bash
# For backend
docker build -f backend/Dockerfile.fastapi -t my-fastapi-app ./backend

# For frontend
docker build -f frontend/Dockerfile.streamlit -t my-streamlit-app ./frontend

# -f specifies the filename
# -t specifies the image name + tag
# . is the build context (should match the folder)

```
ds


