import os
import time
import subprocess
import hashlib
import logging
import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class SimpleJobManager:
    def __init__(self, job_file_path="jobs.txt", done_jobs_path="done_jobs.txt", logs_dir="job_logs"):
        self.job_file_path = job_file_path
        self.done_jobs_path = done_jobs_path
        self.logs_dir = logs_dir
        self.last_file_hash = ""
        
        # Create required directories and files
        if not os.path.exists(self.done_jobs_path):
            open(self.done_jobs_path, 'w').close()
            
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
    
    def _get_file_hash(self):
        """Get hash of the job file to detect changes"""
        if not os.path.exists(self.job_file_path):
            return ""
        with open(self.job_file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_done_jobs(self):
        """Load list of completed jobs"""
        with open(self.done_jobs_path, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    
    def _mark_job_done(self, job):
        """Mark job as completed"""
        with open(self.done_jobs_path, 'a') as f:
            f.write(f"{job}\n")
    
    def _get_pending_jobs(self):
        """Get jobs that need to be run"""
        if not os.path.exists(self.job_file_path):
            return []
            
        done_jobs = self._get_done_jobs()
        
        with open(self.job_file_path, 'r') as f:
            all_jobs = [line.strip() for line in f if line.strip()]
        
        return [job for job in all_jobs if job not in done_jobs]
    
    def _create_log_filename(self, job):
        """Create a meaningful log filename from the job command"""
        # Remove dangerous characters and replace spaces
        safe_name = job.replace("/", "_").replace("\\", "_").replace(" ", "_")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{safe_name[:50]}.log"  # Limit filename length
    
    def _run_job(self, job):
        """Run a single job with output redirected to a log file"""
        log_file = os.path.join(self.logs_dir, self._create_log_filename(job))
        logging.info(f"Starting job: {job}")
        logging.info(f"Output will be logged to: {log_file}")
        
        try:
            # Open log file for writing
            with open(log_file, 'w') as f_log:
                # Write header to log file
                f_log.write(f"=== Job: {job} ===\n")
                f_log.write(f"=== Started: {datetime.datetime.now()} ===\n\n")
                f_log.flush()  # Ensure header is written
                
                # Start process with output redirected to log file
                process = subprocess.Popen(
                    job, 
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                    universal_newlines=True,
                    bufsize=1  # Line buffered
                )
                
                # Stream output to log file in real-time
                for line in process.stdout:
                    f_log.write(line)
                    f_log.flush()  # Ensure output is written immediately
                
                # Wait for process to complete
                return_code = process.wait()
                
                # Write footer to log file
                f_log.write(f"\n=== Finished: {datetime.datetime.now()} ===\n")
                f_log.write(f"=== Return code: {return_code} ===\n")
            
            if return_code == 0:
                logging.info(f"Job completed successfully: {job}")
                self._mark_job_done(job)
                return True
            else:
                logging.error(f"Job failed with exit code {return_code}: {job}")
                return False
                
        except Exception as e:
            logging.error(f"Error running job: {e}")
            # Try to write error to log file
            try:
                with open(log_file, 'a') as f_log:
                    f_log.write(f"\n=== ERROR: {e} ===\n")
            except:
                pass
            return False
    
    def run(self, check_interval=10):
        """Main loop to check for and run jobs"""
        logging.info(f"Starting job manager. Monitoring {self.job_file_path}")
        logging.info(f"Job outputs will be saved to {self.logs_dir}")
        
        while True:
            try:
                # Check if file has changed or we haven't checked it yet
                current_hash = self._get_file_hash()
                file_changed = current_hash != self.last_file_hash
                self.last_file_hash = current_hash
                
                # if file_changed or not self.last_file_hash:
                # Get next job to run
                pending_jobs = self._get_pending_jobs()
                
                if pending_jobs:
                    next_job = pending_jobs[0]
                    self._run_job(next_job)
                else:
                    logging.info("No pending jobs. Waiting for new jobs...")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logging.info("Shutting down...")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(check_interval)

if __name__ == "__main__":
    manager = SimpleJobManager()
    manager.run()