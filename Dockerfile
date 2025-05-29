# Step 1: Use an official Python runtime as a base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file into the container
COPY requirement.txt .

# Step 4: Install the required Python packages
RUN pip install --no-cache-dir -r requirement.txt

# Step 5: Copy the entire project into the container
COPY . .


# Step 6: Expose the port that Streamlit will run on (default port for Streamlit is 8501)
EXPOSE 8000

# Step 7: Command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]
