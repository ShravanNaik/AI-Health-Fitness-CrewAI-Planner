import sys
import importlib
importlib.import_module('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
import time
from datetime import datetime
import json

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🏋️‍♂️ AI Health & Fitness CrewAI Planner",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)


hide_footer_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* This targets GitHub icon in the footer */
    .st-emotion-cache-1y4p8pa.ea3mdgi1 {
        display: none !important;
    }

    /* This targets the entire footer area */
    .st-emotion-cache-164nlkn {
        display: none !important;
    }
    </style>
"""

st.markdown(hide_footer_style, unsafe_allow_html=True)

# Custom CSS for beautiful presentation
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.8rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-5px);
    }
    
    .nutrition-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.8rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .fitness-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.8rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .wellness-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.8rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .profile-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .status-working {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        animation: pulse 2s infinite;
        margin: 0.5rem 0;
    }
    
    .status-complete {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .qa-section {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: #333;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .stExpander > div:first-child {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
    }
    
    .demo-banner {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    .success-highlight {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'health_results' not in st.session_state:
    st.session_state.health_results = None
if 'agent_status' not in st.session_state:
    st.session_state.agent_status = {
        'nutrition': 'pending',
        'fitness': 'pending',
        'wellness': 'pending'
    }
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []
if 'plans_generated' not in st.session_state:
    st.session_state.plans_generated = False

# Demo banner
st.markdown("""
<div class="demo-banner">
    🎭 LIVE DEMO: AI Health & Fitness CrewAI System - Multi-Agent Wellness Planning
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏋️‍♂️ AI Health & Fitness CrewAI Planner</h1>
    <p>Advanced Multi-Agent System for Personalized Health & Wellness</p>
    <p><em>Powered by CrewAI • Nutrition Science • Fitness Expertise • Wellness Coaching</em></p>
</div>
""", unsafe_allow_html=True)

# Check for OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("🔑 OpenAI API key not found! Please set OPENAI_API_KEY in your .env file")
    st.code("""
    # Create a .env file in your project directory with:
    OPENAI_API_KEY=your_openai_api_key_here
    """)
    st.stop()

# Initialize OpenAI
try:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=openai_api_key)
    st.success("🤖 AI Crew Initialized Successfully!")
except Exception as e:
    st.error(f"❌ Error initializing AI models: {str(e)}")
    st.stop()

# Sidebar with agent status
with st.sidebar:
    st.markdown("### 🤖 Agent Status Dashboard")
    
    status_colors = {
        'pending': '⏳',
        'working': '🔄',
        'complete': '✅'
    }
    
    for agent, status in st.session_state.agent_status.items():
        if status == 'working':
            st.markdown(f"""
            <div class="status-working">
                <strong>{agent.title()} Specialist:</strong> {status_colors[status]} {status.title()}
            </div>
            """, unsafe_allow_html=True)
        elif status == 'complete':
            st.markdown(f"""
            <div class="status-complete">
                <strong>{agent.title()} Specialist:</strong> {status_colors[status]} {status.title()}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"**{agent.title()} Specialist:** {status_colors[status]} {status.title()}")
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Crew Analytics
    - **Total Agents:** 3
    - **Specializations:** Nutrition, Fitness, Wellness
    - **Process:** Sequential Collaboration
    - **AI Model:** GPT-4o
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### ⚠️ Health Disclaimer
    This AI system provides educational guidance and should not replace professional medical advice. Always consult healthcare providers for medical decisions.
    """)

# Agent showcase
st.markdown("### 🤖 Meet Your Health & Fitness AI Crew")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="nutrition-card">
        <h4>🥗 Nutrition Specialist</h4>
        <p><strong>Dr. Maria Rodriguez, AI</strong></p>
        <p>• Personalized meal planning<br>
        • Macronutrient optimization<br>
        • Dietary restriction management<br>
        • Nutritional goal alignment<br>
        • Evidence-based recommendations</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="fitness-card">
        <h4>💪 Fitness Trainer</h4>
        <p><strong>Coach Alex Thompson, AI</strong></p>
        <p>• Custom workout design<br>
        • Progressive training plans<br>
        • Exercise form guidance<br>
        • Goal-specific routines<br>
        • Performance optimization</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="wellness-card">
        <h4>🧘 Wellness Coach</h4>
        <p><strong>Dr. Sarah Kim, AI</strong></p>
        <p>• Holistic lifestyle planning<br>
        • Recovery optimization<br>
        • Stress management<br>
        • Sleep quality improvement<br>
        • Mind-body integration</p>
    </div>
    """, unsafe_allow_html=True)

# User profile input
st.markdown("### 👤 Your Health Profile")

st.markdown("""
<div class="profile-section">
    <h4>📝 Personal Information</h4>
    <p>Please provide accurate information for the most personalized recommendations</p>
</div>
""", unsafe_allow_html=True)

with st.form("health_profile"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Physical Metrics")
        age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.1)
        sex = st.selectbox("Sex", options=["Male", "Female", "Other"])
        
        st.markdown("#### 🎯 Goals & Preferences")
        fitness_goals = st.selectbox(
            "Primary Fitness Goal",
            options=["Lose Weight", "Gain Muscle", "Build Endurance", "General Fitness", 
                    "Strength Training", "Athletic Performance", "Rehabilitation"]
        )
        
        dietary_preferences = st.selectbox(
            "Dietary Preference",
            options=["No Restrictions", "Vegetarian", "Vegan", "Keto", "Paleo", 
                    "Mediterranean", "Low Carb", "Gluten Free", "Dairy Free"]
        )
    
    with col2:
        st.markdown("#### 🏃‍♂️ Lifestyle Factors")
        activity_level = st.selectbox(
            "Current Activity Level",
            options=["Sedentary (desk job, no exercise)", 
                    "Lightly Active (light exercise 1-3 days/week)",
                    "Moderately Active (moderate exercise 3-5 days/week)",
                    "Very Active (hard exercise 6-7 days/week)",
                    "Extremely Active (physical job + exercise)"]
        )
        
        workout_experience = st.selectbox(
            "Workout Experience",
            options=["Beginner (0-6 months)", "Intermediate (6 months - 2 years)", 
                    "Advanced (2+ years)", "Expert (5+ years)"]
        )
        
        time_availability = st.selectbox(
            "Available Workout Time",
            options=["15-30 minutes", "30-45 minutes", "45-60 minutes", 
                    "60-90 minutes", "90+ minutes"]
        )
        
        health_conditions = st.multiselect(
            "Health Considerations (if any)",
            options=["None", "Diabetes", "Hypertension", "Heart Disease", 
                    "Joint Issues", "Back Problems", "Injuries", "Other"]
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 🍎 Nutrition Details")
        current_diet_quality = st.slider("Current Diet Quality (1-10)", 1, 10, 5)
        daily_water_intake = st.slider("Daily Water Intake (glasses)", 1, 15, 8)
        meal_prep_time = st.selectbox(
            "Available Meal Prep Time",
            options=["Minimal (quick meals)", "Moderate (30-60 min/day)", 
                    "Extensive (1+ hours/day)"]
        )
    
    with col4:
        st.markdown("#### 😴 Wellness Factors")
        sleep_hours = st.slider("Average Sleep Hours", 4, 12, 7)
        stress_level = st.slider("Current Stress Level (1-10)", 1, 10, 5)
        energy_level = st.slider("Daily Energy Level (1-10)", 1, 10, 6)
    
    # Advanced preferences
    with st.expander("🔧 Advanced Preferences"):
        budget_range = st.selectbox(
            "Nutrition Budget Range",
            options=["Budget-friendly", "Moderate", "Premium", "No constraints"]
        )
        
        equipment_access = st.multiselect(
            "Available Equipment",
            options=["None (bodyweight only)", "Basic (dumbbells, resistance bands)", 
                    "Home gym", "Commercial gym", "Outdoor spaces"]
        )
        
        preferred_workout_style = st.multiselect(
            "Preferred Workout Styles",
            options=["Strength training", "Cardio", "HIIT", "Yoga", "Pilates", 
                    "Swimming", "Running", "Cycling", "Sports"]
        )
    
    submit_button = st.form_submit_button(
        "🚀 Activate Health & Fitness AI Crew",
        use_container_width=True
    )

# Process the health assessment
if submit_button:
    # Create comprehensive user profile
    user_profile = f"""
    COMPREHENSIVE HEALTH & FITNESS PROFILE:
    
    PHYSICAL METRICS:
    - Age: {age} years
    - Height: {height} cm
    - Weight: {weight} kg
    - Sex: {sex}
    - BMI: {round(weight / ((height/100) ** 2), 1)}
    
    GOALS & PREFERENCES:
    - Primary Goal: {fitness_goals}
    - Dietary Preference: {dietary_preferences}
    - Activity Level: {activity_level}
    - Workout Experience: {workout_experience}
    - Time Availability: {time_availability}
    
    HEALTH CONSIDERATIONS:
    - Health Conditions: {', '.join(health_conditions)}
    - Current Diet Quality: {current_diet_quality}/10
    - Daily Water Intake: {daily_water_intake} glasses
    - Sleep Hours: {sleep_hours} hours
    - Stress Level: {stress_level}/10
    - Energy Level: {energy_level}/10
    
    PREFERENCES:
    - Meal Prep Time: {meal_prep_time}
    - Budget Range: {budget_range}
    - Equipment Access: {', '.join(equipment_access) if equipment_access else 'None specified'}
    - Preferred Workout Styles: {', '.join(preferred_workout_style) if preferred_workout_style else 'Open to suggestions'}
    
    Assessment Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    """
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    
    with st.spinner("🤖 Initializing Health & Fitness AI Crew..."):
        
        # Define CrewAI agents
        nutrition_agent = Agent(
            role="Registered Dietitian and Nutrition Specialist",
            goal="Create comprehensive, personalized nutrition plans that align with health goals, dietary preferences, and lifestyle factors",
            backstory="""You are Dr. Maria Rodriguez, a registered dietitian with 12 years of experience in clinical nutrition 
            and sports dietetics. You specialize in creating evidence-based, sustainable nutrition plans that consider individual 
            metabolic needs, food preferences, and lifestyle constraints. Your approach combines nutritional science with 
            practical meal planning, ensuring clients can maintain their nutrition goals long-term.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3
        )
        
        fitness_agent = Agent(
            role="Certified Personal Trainer and Exercise Physiologist",
            goal="Design safe, effective, and progressive fitness programs tailored to individual goals, experience levels, and physical capabilities",
            backstory="""You are Coach Alex Thompson, a certified personal trainer and exercise physiologist with 10 years 
            of experience training clients from beginners to elite athletes. You excel at creating periodized training programs 
            that maximize results while minimizing injury risk. Your expertise spans strength training, cardiovascular conditioning, 
            functional movement, and sport-specific training.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3
        )
        
        wellness_agent = Agent(
            role="Holistic Wellness Coach and Lifestyle Optimization Specialist",
            goal="Integrate nutrition and fitness plans into a comprehensive wellness strategy that promotes sustainable health habits and optimal life balance",
            backstory="""You are Dr. Sarah Kim, a holistic wellness coach with expertise in integrative health, stress management, 
            and lifestyle medicine. With 8 years of experience, you specialize in helping clients create sustainable wellness 
            routines that address the interconnected aspects of health: nutrition, movement, sleep, stress management, and 
            mental well-being. Your approach emphasizes gradual, sustainable changes that fit into real-life schedules.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            max_iter=3
        )
        
        # Define tasks
        nutrition_task = Task(
            description=f"""
            Create an EXCITING, ENGAGING, and VISUALLY APPEALING personalized nutrition plan that makes the user excited to start their health journey!
            
            {user_profile}
            
            Your nutrition plan must be formatted like a PREMIUM PRODUCT with:
            
            1. **EXCITING DAILY VARIETY** - Create 7 COMPLETELY DIFFERENT daily meal plans with:
               - Creative breakfast combinations (smoothie bowls, overnight oats variations, protein pancakes, etc.)
               - Diverse lunch options (Buddha bowls, wraps, salads, soups)
               - Exciting dinner themes (Mediterranean Monday, Taco Tuesday, Asian-inspired Wednesday, etc.)
               - Fun snack combinations that feel like treats
            
            2. **PREMIUM PRESENTATION** with:
               - Appetizing meal descriptions that make food sound irresistible
               - Colorful ingredient combinations
               - Chef-style cooking tips and tricks
               - "Why you'll love this" explanations for each meal
            
            3. **INTERACTIVE ELEMENTS**:
               - Weekly themes and challenges
               - "Swap options" for flexibility
               - "Power-up additions" for extra nutrition
               - "Quick hacks" for busy days
            
            4. **MOTIVATIONAL LANGUAGE**:
               - Use exciting, positive language
               - Frame foods as "fuel," "energy boosters," "metabolism igniters"
               - Include empowering statements about each meal choice
            
            5. **PRACTICAL MAGIC**:
               - Smart meal prep strategies that save time
               - "Batch cooking" power hours
               - Emergency backup meal options
               - Restaurant ordering guides
            
            Format like a premium nutrition guide that users would PAY FOR, not a boring clinical plan!
            Make every meal sound delicious and achievable!
            """,
            agent=nutrition_agent,
            expected_output="An exciting, premium-quality nutrition plan (1000-1200 words) with varied daily menus, engaging descriptions, and motivational presentation that makes healthy eating irresistible."
        )
        
        fitness_task = Task(
            description=f"""
            Design an EXCITING, DYNAMIC fitness program that makes working out feel like an adventure, not a chore!
            
            {user_profile}
            
            Create a fitness program that feels like a PREMIUM PERSONAL TRAINING EXPERIENCE:
            
            1. **THEMED WORKOUT DAYS** with exciting names:
               - "Warrior Wednesday" (strength focus)
               - "Flow Friday" (yoga and mobility)
               - "Cardio Carnival" (fun cardio circuits)
               - "Power-Up Monday" (full body energy)
            
            2. **PROGRESSIVE CHALLENGES** that feel like gaming:
               - Weekly fitness challenges with rewards
               - "Level up" progressions that feel achievable
               - Personal records to beat
               - Milestone celebrations
            
            3. **VARIETY AND FUN**:
               - 4-5 different workout styles per week
               - Music playlist suggestions for each workout type
               - "Workout of the day" alternatives
               - Partner workout options for social motivation
            
            4. **SMART ADAPTATIONS**:
               - "Busy day" 10-minute express versions
               - "Energized day" extended challenges
               - Equipment-free alternatives
               - Indoor/outdoor options
            
            5. **MOTIVATIONAL PRESENTATION**:
               - Inspiring workout descriptions
               - "What this does for you" benefits for each exercise
               - Success visualization techniques
               - Progress celebration milestones
            
            6. **RECOVERY AS LUXURY**:
               - Spa-style recovery routines
               - Meditation and breathwork integration
               - Self-massage techniques
               - Active recovery adventures
            
            Make fitness feel like the BEST PART of their day, not a burden!
            Present like a premium fitness coaching program worth hundreds of dollars!
            """,
            agent=fitness_agent,
            expected_output="An exciting, premium fitness program (1000-1200 words) with themed workouts, progressive challenges, and motivational presentation that makes exercise irresistible.",
            context=[nutrition_task]
        )
        
        wellness_task = Task(
            description=f"""
            Create an INSPIRING, LUXURY LIFESTYLE TRANSFORMATION plan that makes wellness feel like a premium lifestyle upgrade!
            
            {user_profile}
            
            Design a wellness strategy that feels like a LIFE MAKEOVER PROGRAM:
            
            1. **MORNING RITUALS** that feel like self-care luxury:
               - 5-minute morning energy rituals
               - Mindfulness practices that boost confidence
               - Hydration ceremonies with lemon water or herbal teas
               - Power affirmations for daily motivation
            
            2. **EVENING WIND-DOWN** like a spa experience:
               - Relaxing evening routines
               - Stress-melting techniques
               - Digital detox strategies
               - Sleep optimization secrets
            
            3. **LIFESTYLE HACKS** for busy professionals:
               - Desk exercise mini-breaks
               - Healthy snack strategies for work
               - Energy management throughout the day
               - Productivity boosters through nutrition
            
            4. **SOCIAL & EMOTIONAL WELLNESS**:
               - Building supportive communities
               - Confidence-building activities
               - Celebration rituals for achievements
               - Stress transformation techniques
            
            5. **HABIT STACKING** like a productivity guru:
               - Simple habit chains that stick
               - "If this, then that" wellness triggers
               - Micro-habits that create big changes
               - Environmental design for success
            
            6. **PROGRESS TRACKING** like a life coach:
               - Weekly check-in rituals
               - Energy and mood tracking
               - Photo progress celebrations
               - Goal evolution strategies
            
            7. **OBSTACLE TRANSFORMATION**:
               - Turn challenges into opportunities
               - Backup plans that feel exciting
               - Flexible strategies for real life
               - Self-compassion practices
            
            Present this as a COMPLETE LIFESTYLE TRANSFORMATION PROGRAM that addresses mind, body, and spirit!
            Make wellness feel like an exciting journey of self-discovery and empowerment!
            """,
            agent=wellness_agent,
            expected_output="An inspiring lifestyle transformation plan (1000-1200 words) with luxury wellness rituals, practical life hacks, and empowering strategies that make healthy living irresistible.",
            context=[nutrition_task, fitness_task]
        )
        
        # Create and execute crew
        health_crew = Crew(
            agents=[nutrition_agent, fitness_agent, wellness_agent],
            tasks=[nutrition_task, fitness_task, wellness_task],
            process=Process.sequential,
            verbose=True
        )
        
        # Execute with progress updates
        progress_bar.progress(25)
        status_placeholder.markdown("🥗 **Nutrition Specialist** creating your personalized meal plan...")
        st.session_state.agent_status['nutrition'] = 'working'
        
        time.sleep(1)
        
        progress_bar.progress(50)
        status_placeholder.markdown("💪 **Fitness Trainer** designing your workout program...")
        st.session_state.agent_status['nutrition'] = 'complete'
        st.session_state.agent_status['fitness'] = 'working'
        
        time.sleep(1)
        
        progress_bar.progress(75)
        status_placeholder.markdown("🧘 **Wellness Coach** integrating your lifestyle plan...")
        st.session_state.agent_status['fitness'] = 'complete'
        st.session_state.agent_status['wellness'] = 'working'
        
        try:
            result = health_crew.kickoff()
            
            progress_bar.progress(100)
            status_placeholder.markdown("✅ **All Specialists Complete** - Your personalized health plan is ready!")
            st.session_state.agent_status['wellness'] = 'complete'
            
            st.session_state.health_results = result
            st.session_state.plans_generated = True
            st.session_state.qa_pairs = []
            
            time.sleep(1)
            status_placeholder.empty()
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"❌ An error occurred during crew execution: {str(e)}")

# Display results
if st.session_state.health_results:
    st.markdown("---")
    st.markdown("## 📋 Your Personalized Health & Fitness Plan")
    
    try:
        tasks_output = st.session_state.health_results.tasks_output
        
        # Display nutrition plan
        with st.expander("🥗 **Your Premium Nutrition Transformation**", expanded=True):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                       padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
                <h3>🌟 Your Personalized Meal Mastery Plan</h3>
                <p>Designed specifically for your goals, preferences, and lifestyle!</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(tasks_output[0].raw)
        
        # Display fitness program
        with st.expander("💪 **Your Elite Fitness Adventure**", expanded=True):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                       padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
                <h3>🔥 Your Custom Training Experience</h3>
                <p>Workouts designed to energize, strengthen, and transform your body!</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(tasks_output[1].raw)
        
        # Display wellness integration
        with st.expander("🧘 **Your Complete Lifestyle Transformation**", expanded=True):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                       padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 1rem;">
                <h3>✨ Your Holistic Wellness Journey</h3>
                <p>Mind, body, and lifestyle integration for lasting transformation!</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(tasks_output[2].raw)
        
        # Success metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🥗 Nutrition Plan</h4>
                <p>✅ 7 Unique Daily Menus</p>
                <p>🌟 Premium Quality</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>💪 Fitness Program</h4>
                <p>✅ Themed Workouts</p>
                <p>🔥 Maximum Engagement</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>🧘 Lifestyle Plan</h4>
                <p>✅ Complete Transformation</p>
                <p>✨ Holistic Integration</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-highlight">
            <h3>🎉 CONGRATULATIONS! Your Premium Health Transformation Package is Ready!</h3>
            <p>🌟 <strong>WHAT YOU GET:</strong> 7 unique daily meal plans + themed workouts + complete lifestyle makeover</p>
            <p>💎 <strong>VALUE:</strong> This premium package would cost $300+ with a personal trainer and nutritionist</p>
            <p>🚀 <strong>RESULTS:</strong> Start seeing changes in energy within 3 days, body changes within 2 weeks!</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

# Q&A Section
if st.session_state.plans_generated:
    st.markdown("---")
    st.markdown("""
    <div class="qa-section">
        <h3>💬 Got Questions? Your AI Experts Are Here!</h3>
        <p>🎯 <strong>Ask anything!</strong> Our specialized AI crew knows your plan inside and out</p>
        <p>💡 <strong>Popular questions:</strong> "Can I swap this meal?" • "What if I miss a workout?" • "How do I stay motivated?"</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("qa_form"):
        question = st.text_area(
            "💭 What's on your mind?",
            placeholder="🔥 POWER QUESTIONS TO TRY:\n\n• Can I make this meal in 10 minutes?\n• What if I hate quinoa? Give me alternatives!\n• I'm traveling next week - help me stay on track!\n• How do I make this workout more challenging?\n• What should I do if I plateau?\n• Can my family eat these meals too?",
            height=120
        )
        
        ask_button = st.form_submit_button("🧠 Get Your Expert Answer Now!", use_container_width=True)
    
    if ask_button and question:
        with st.spinner("🧠 AI Crew analyzing your question..."):
            try:
                # Create a Q&A agent
                qa_agent = Agent(
                    role="Health & Fitness Q&A Specialist",
                    goal="Provide expert answers to questions about the user's personalized health and fitness plan",
                    backstory="""You are a knowledgeable health and fitness expert who has access to the user's 
                    complete personalized plan. You provide clear, actionable answers that help users understand 
                    and implement their plans successfully.""",
                    verbose=True,
                    allow_delegation=False,
                    llm=llm,
                    max_iter=2
                )
                
                # Create context from existing plans
                if st.session_state.health_results:
                    plans_context = "\n\n".join([task.raw for task in st.session_state.health_results.tasks_output])
                    
                    qa_task = Task(
                        description=f"""
                        Answer the user's question based on their personalized health and fitness plan:
                        
                        USER'S QUESTION: {question}
                        
                        PERSONALIZED PLANS CONTEXT:
                        {plans_context}
                        
                        Provide a clear, helpful answer that:
                        1. Directly addresses their question
                        2. References their specific plan when relevant
                        3. Offers practical alternatives or modifications if needed
                        4. Maintains safety and health considerations
                        5. Encourages adherence to their goals
                        """,
                        agent=qa_agent,
                        expected_output="A clear, helpful answer (200-400 words) that directly addresses the user's question with practical guidance."
                    )
                    
                    qa_crew = Crew(
                        agents=[qa_agent],
                        tasks=[qa_task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    answer_result = qa_crew.kickoff()
                    answer = answer_result.tasks_output[0].raw
                    
                    st.session_state.qa_pairs.append((question, answer))
                
            except Exception as e:
                st.error(f"❌ Error getting answer: {str(e)}")
    
    # Display Q&A history
    if st.session_state.qa_pairs:
        st.markdown("### 💬 Q&A History")
        for i, (q, a) in enumerate(reversed(st.session_state.qa_pairs)):
            with st.expander(f"Q{len(st.session_state.qa_pairs)-i}: {q[:50]}..."):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")

# Download section
if st.session_state.health_results:
    st.markdown("### 💾 Save Your Complete Plan")
    
    # Create downloadable content
    plan_content = f"""
    COMPLETE HEALTH & FITNESS PLAN
    Generated by AI Health & Fitness CrewAI System
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    
    ==========================================
    PERSONALIZED NUTRITION PLAN
    ==========================================
    {st.session_state.health_results.tasks_output[0].raw}
    
    ==========================================
    CUSTOM FITNESS PROGRAM
    ==========================================
    {st.session_state.health_results.tasks_output[1].raw}
    
    ==========================================
    HOLISTIC WELLNESS STRATEGY
    ==========================================
    {st.session_state.health_results.tasks_output[2].raw}
    
    ==========================================
    Q&A HISTORY
    ==========================================
    """
    
    if st.session_state.qa_pairs:
        for i, (q, a) in enumerate(st.session_state.qa_pairs, 1):
            plan_content += f"""
    Q{i}: {q}
    A{i}: {a}
    
    """
    
    plan_content += """
    ==========================================
    DISCLAIMER
    ==========================================
    This plan was generated by AI and should complement, not replace, professional medical advice.
    Always consult healthcare providers for medical decisions and before starting new health programs.
    """
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download Complete Plan (Text Format)",
            data=plan_content,
            file_name=f"health_fitness_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        # Create a summary version
        summary_content = f"""
        HEALTH & FITNESS PLAN SUMMARY
        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        
        🎯 PRIMARY GOAL: {fitness_goals}
        🥗 DIET TYPE: {dietary_preferences}
        💪 FITNESS LEVEL: {workout_experience}
        ⏰ WORKOUT TIME: {time_availability}
        
        KEY RECOMMENDATIONS:
        • Follow personalized nutrition plan with calculated macros
        • Complete structured workout program {time_availability}
        • Implement wellness strategies for sustainable results
        • Track progress and adjust as needed
        
        Generated by AI Health & Fitness CrewAI System
        """
        
        st.download_button(
            label="📄 Download Quick Summary",
            data=summary_content,
            file_name=f"health_plan_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Footer for demo
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px;">
    <h3>🎭 Health & Fitness CrewAI Demo Complete</h3>
    <p><strong>Multi-Agent AI System for Personalized Wellness</strong></p>
    <p>Nutrition Science • Fitness Expertise • Wellness Coaching • Seamless Integration</p>
    <p><em>Powered by CrewAI • Built for Scale • Designed for Results</em></p>
</div>
""", unsafe_allow_html=True)

# Technical specifications for demo
with st.expander("🔧 Technical Specifications (Demo Info)"):
    st.markdown("""
    ### CrewAI Architecture
    - **Framework**: CrewAI with Sequential Process
    - **AI Model**: OpenAI GPT-4o
    - **Agent Count**: 3 specialized agents
    - **Task Coordination**: Context-aware sequential execution
    - **API Management**: Environment variables with dotenv
    
    ### Agent Specializations
    1. **Nutrition Specialist**: Registered Dietitian persona with 12 years experience
    2. **Fitness Trainer**: Certified trainer and exercise physiologist with 10 years experience  
    3. **Wellness Coach**: Holistic wellness expert with 8 years integrative health experience
    
    ### Features Demonstrated
    - Multi-agent collaboration and context sharing
    - Sequential task processing with dependency management
    - Real-time progress tracking and status updates
    - Interactive Q&A system with plan-specific context
    - Professional UI/UX optimized for presentations
    - Comprehensive plan generation and download capabilities
    """)
