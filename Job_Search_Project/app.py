from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import os
import openai
import json
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv('config.env')

app = Flask(__name__)
CORS(app)

# Configure Google Gemini
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Openai Model

# openai.api_key = os.getenv("OPENAI_API_KEY")

# Example call for OpenAI GPT-4o
# response = openai.ChatCompletion.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Your prompt here"}
#     ]
# )


# RapidAPI Configuration
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY')
RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST')

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/search-jobs', methods=['GET'])
def search_jobs():
    """Search for jobs using RapidAPI and enhance with Gemini AI"""
    try:
        # Get query parameters
        query = request.args.get('q', '')
        location = request.args.get('location', '')
        date_posted = request.args.get('date_posted', 'all')
        country = request.args.get('country', 'us')
        page = request.args.get('page', '1')
        num_pages = request.args.get('num_pages', '1')
        filter_date_range = request.args.get('filter_date_range', '')  # New parameter for date filtering
        
        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400
        
        # Build search query
        search_query = query
        if location:
            search_query += f" in {location}"
        
        # Search jobs using RapidAPI
        jobs_data = search_jobs_rapidapi(search_query, country, date_posted, page, num_pages)
        
        if not jobs_data or 'data' not in jobs_data:
            return jsonify({'jobs': [], 'message': 'No jobs found'})
        
        # Process and enhance job results with Gemini AI (with timeout protection)
        try:
            enhanced_jobs = enhance_jobs_with_gemini(jobs_data['data'], query)
        except Exception as ai_error:
            print(f"AI enhancement failed, using raw data: {ai_error}")
            # Fallback to raw job data without AI enhancement
            enhanced_jobs = []
            for job in jobs_data['data'][:5]:  # Limit to 5 jobs
                enhanced_jobs.append({
                    'title': job.get('job_title', 'N/A'),
                    'company': job.get('employer_name', 'N/A'),
                    'location': f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(', '),
                    'link': job.get('job_apply_link', ''),
                    'relevance_score': 'medium',
                    'summary': 'Job details available',
                    'original_link': job.get('job_apply_link', ''),
                    'posted_date': job.get('job_posted_at_datetime_utc', '')
                })
        
        # Apply date range filter if specified
        if filter_date_range:
            current_date = datetime.now()
            if filter_date_range == '7_days':
                cutoff_date = current_date - timedelta(days=7)
            elif filter_date_range == '30_days':
                cutoff_date = current_date - timedelta(days=30)
            else:
                cutoff_date = None
            
            if cutoff_date:
                enhanced_jobs = [job for job in enhanced_jobs 
                               if job.get('posted_date') and 
                               is_job_recent(job['posted_date'], cutoff_date)]
        
        # Don't limit results initially - show all found jobs
        # enhanced_jobs = enhanced_jobs[:5]  # Removed initial limit
        
        return jsonify({
            'jobs': enhanced_jobs,
            'total_results': len(enhanced_jobs),
            'query': query,
            'location': location,
            'date_filter': filter_date_range
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': 'Search failed. Please try again.'}), 500

def is_job_recent(posted_date_str, cutoff_date):
    """Helper function to compare job posting date with cutoff date"""
    try:
        # Handle different date formats from RapidAPI
        if posted_date_str.endswith('Z'):
            # UTC format: "2024-01-15T10:30:00Z"
            posted_date = datetime.fromisoformat(posted_date_str.replace('Z', '+00:00'))
        elif '+' in posted_date_str:
            # ISO format with timezone: "2024-01-15T10:30:00+00:00"
            posted_date = datetime.fromisoformat(posted_date_str)
        else:
            # Naive datetime: "2024-01-15T10:30:00"
            posted_date = datetime.fromisoformat(posted_date_str)
            # Assume UTC if no timezone info
            posted_date = posted_date.replace(tzinfo=None)
        
        # Make cutoff_date timezone-aware if posted_date is timezone-aware
        if posted_date.tzinfo is not None and cutoff_date.tzinfo is None:
            cutoff_date = cutoff_date.replace(tzinfo=None)
        
        return posted_date > cutoff_date
    except Exception as e:
        print(f"Error parsing date {posted_date_str}: {e}")
        return True  # Include job if date parsing fails

def search_jobs_rapidapi(query, country, date_posted, page, num_pages):
    """Search for jobs using RapidAPI JSearch"""
    url = "https://jsearch.p.rapidapi.com/search"
    
    querystring = {
        "query": query,
        "page": page,
        "num_pages": num_pages,
        "country": country,
        "date_posted": date_posted
    }
    
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=30)  # 30 second timeout
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("RapidAPI request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"RapidAPI Error: {e}")
        return None

def enhance_jobs_with_gemini(jobs_data, user_query):
    """Enhance job results using Google Gemini AI"""
    enhanced_jobs = []
    
    # Limit the number of jobs to process to avoid timeouts
    jobs_to_process = jobs_data[:3]  # Only process first 3 jobs to avoid timeouts
    
    for i, job in enumerate(jobs_to_process):
        try:
            # Create a simpler, faster prompt for Gemini
            prompt = f"""
            Job: {job.get('job_title', 'N/A')} at {job.get('employer_name', 'N/A')}
            Location: {job.get('job_city', 'N/A')}, {job.get('job_country', 'N/A')}
            User Query: {user_query}
            
            Return JSON: {{"title": "job title", "company": "company", "location": "location", "link": "{job.get('job_apply_link', '')}", "relevance_score": "high/medium/low", "summary": "brief summary"}}
            """
            
            # Get Gemini response (removed timeout parameter as it's not supported)
            try:
                response = model.generate_content(prompt)
                
                # Parse Gemini response
                try:
                    # Extract JSON from response
                    response_text = response.text
                    if '```json' in response_text:
                        json_start = response_text.find('```json') + 7
                        json_end = response_text.find('```', json_start)
                        json_str = response_text[json_start:json_end].strip()
                    else:
                        # Try to find JSON in the response
                        start_idx = response_text.find('{')
                        end_idx = response_text.rfind('}') + 1
                        json_str = response_text[start_idx:end_idx]
                    
                    enhanced_job = json.loads(json_str)
                    
                    # Add original data as fallback
                    enhanced_job['original_link'] = job.get('job_apply_link', '')
                    enhanced_job['posted_date'] = job.get('job_posted_at_datetime_utc', '')
                    
                    enhanced_jobs.append(enhanced_job)
                    
                except json.JSONDecodeError:
                    # Fallback if Gemini doesn't return valid JSON
                    enhanced_jobs.append({
                        'title': job.get('job_title', 'N/A'),
                        'company': job.get('employer_name', 'N/A'),
                        'location': f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(', '),
                        'link': job.get('job_apply_link', ''),
                        'relevance_score': 'medium',
                        'summary': 'Job details available',
                        'original_link': job.get('job_apply_link', ''),
                        'posted_date': job.get('job_posted_at_datetime_utc', '')
                    })
                    
            except Exception as gemini_error:
                print(f"Gemini API error for job {i+1}: {gemini_error}")
                # Fallback to original data
                enhanced_jobs.append({
                    'title': job.get('job_title', 'N/A'),
                    'company': job.get('employer_name', 'N/A'),
                    'location': f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(', '),
                    'link': job.get('job_apply_link', ''),
                    'relevance_score': 'medium',
                    'summary': 'Job details available',
                    'original_link': job.get('job_apply_link', ''),
                    'posted_date': job.get('job_posted_at_datetime_utc', '')
                })
                
        except Exception as e:
            print(f"Error processing job {i+1}: {e}")
            # Fallback to original data
            enhanced_jobs.append({
                'title': job.get('job_title', 'N/A'),
                'company': job.get('employer_name', 'N/A'),
                'location': f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(', '),
                'link': job.get('job_apply_link', ''),
                'relevance_score': 'medium',
                'summary': 'Job details available',
                'original_link': job.get('job_apply_link', ''),
                'posted_date': job.get('job_posted_at_datetime_utc', '')
            })
    
    # Add remaining jobs without AI enhancement if we have more than 3
    if len(jobs_data) > 3:
        for job in jobs_data[3:]:
            enhanced_jobs.append({
                'title': job.get('job_title', 'N/A'),
                'company': job.get('employer_name', 'N/A'),
                'location': f"{job.get('job_city', '')}, {job.get('job_country', '')}".strip(', '),
                'link': job.get('job_apply_link', ''),
                'relevance_score': 'medium',
                'summary': 'Job details available',
                'original_link': job.get('job_apply_link', ''),
                'posted_date': job.get('job_posted_at_datetime_utc', '')
            })
    
    # Sort by relevance score
    relevance_order = {'high': 3, 'medium': 2, 'low': 1}
    enhanced_jobs.sort(key=lambda x: relevance_order.get(x.get('relevance_score', 'medium'), 1), reverse=True)
    
    return enhanced_jobs

@app.route('/api/filter-jobs', methods=['POST'])
def filter_jobs():
    """Filter jobs based on criteria"""
    try:
        data = request.get_json()
        jobs = data.get('jobs', [])
        filters = data.get('filters', {})
        
        filtered_jobs = jobs
        
        # Filter by location
        if filters.get('location'):
            location_filter = filters['location'].lower()
            filtered_jobs = [job for job in filtered_jobs 
                           if location_filter in job.get('location', '').lower()]
        
        # Filter by relevance score
        if filters.get('relevance_score'):
            score_filter = filters['relevance_score'].lower()
            filtered_jobs = [job for job in filtered_jobs 
                           if job.get('relevance_score', '').lower() == score_filter]
        
        # Filter by date posted (last 7 days, 30 days, etc.)
        if filters.get('date_range'):
            date_range = filters['date_range']
            current_date = datetime.now()
            
            if date_range == '7_days':
                cutoff_date = current_date - timedelta(days=7)
            elif date_range == '30_days':
                cutoff_date = current_date - timedelta(days=30)
            else:
                cutoff_date = None
            
            if cutoff_date:
                filtered_jobs = [job for job in filtered_jobs 
                               if job.get('posted_date') and 
                               is_job_recent(job['posted_date'], cutoff_date)]
        
        # Filter by top N results
        if filters.get('top_results'):
            try:
                top_n = int(filters['top_results'])
                filtered_jobs = filtered_jobs[:top_n]
            except ValueError:
                pass  # Invalid number, ignore the filter
        
        return jsonify({
            'jobs': filtered_jobs,
            'total_results': len(filtered_jobs)
        })
        
    except Exception as e:
        print(f"Filter error: {e}")
        return jsonify({'error': 'Filter failed. Please try again.'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
