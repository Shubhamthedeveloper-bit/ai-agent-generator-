import tkinter as tk
from tkinter import scrolledtext
from transformers import pipeline
from textblob import TextBlob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Load NLP pipeline for goal analysis
nlp = pipeline("zero-shot-classification",
               model="facebook/bart-large-mnli")

candidate_labels = ['report_generation', 'sentiment_analysis', 'data_analysis', 'social_media_monitoring', 'web_scraping']

# Dummy tweets for sentiment analysis (replace with real API in future)
def get_tweets_dummy():
    return [
        "I love AI and its applications!",
        "AI sometimes scares me with its power.",
        "Data science and AI are the future.",
        "Sentiment analysis is very useful in business.",
        "I am learning to build AI agents."
    ]

# Sentiment analysis using TextBlob
def analyze_sentiment(tweets):
    sentiments = []
    for tweet in tweets:
        blob = TextBlob(tweet)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        sentiments.append((tweet, sentiment, polarity))
    return sentiments

# Data Analysis function: show histogram and summary
def show_data_analysis():
    data = np.random.normal(0, 1, 1000)  # Dummy data

    mean = np.mean(data)
    std_dev = np.std(data)

    # Clear previous chart if any
    for widget in chart_frame.winfo_children():
        widget.destroy()

    summary_text = f"Data Summary:\nMean: {mean:.2f}\nStandard Deviation: {std_dev:.2f}\n"
    output_text.insert(tk.END, "\n" + summary_text + "\n")

    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(data, bins=30, color='skyblue', edgecolor='black')
    ax.set_title("Data Distribution Histogram")

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Global variable to keep last analysis text for PDF report
last_report_text = ""

def analyze_goal():
    global last_report_text
    goal = entry.get()
    if not goal.strip():
        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, "Please enter a goal.")
        return

    result = nlp(goal, candidate_labels)
    tasks = result['labels']
    confidence = result['scores']

    output_text.delete('1.0', tk.END)
    report = "=== Parsed Requirements ===\n"
    report += f"Tasks: {tasks}\n"
    report += f"Confidence: {confidence}\n\n"

    if 'sentiment_analysis' in tasks:
        output_text.insert(tk.END, "Running dummy sentiment analysis on sample tweets...\n\n")
        tweets = get_tweets_dummy()
        analysis_results = analyze_sentiment(tweets)
        for i, (tweet, sentiment, polarity) in enumerate(analysis_results, 1):
            output_text.insert(tk.END, f"{i}. Tweet: {tweet}\n")
            output_text.insert(tk.END, f"   Sentiment: {sentiment}, Polarity Score: {polarity}\n\n")

    if 'data_analysis' in tasks:
        output_text.insert(tk.END, "Performing data analysis and generating chart...\n\n")
        show_data_analysis()

    if 'report_generation' in tasks:
        output_text.insert(tk.END, "You can generate PDF report after running the analysis by clicking 'Generate PDF Report'.\n\n")

    last_report_text = report

# PDF generation function
def generate_pdf_report():
    global last_report_text
    if not last_report_text:
        output_text.insert(tk.END, "\nNo analysis to save. Run the agent first.\n")
        return

    filename = "AI_Agent_Report.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    lines = last_report_text.split('\n')
    y = height - 40
    for line in lines:
        c.drawString(40, y, line)
        y -= 15
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()

    output_text.insert(tk.END, f"\nPDF Report generated and saved as '{filename}'.\n")

# GUI setup
root = tk.Tk()
root.title("Meta-AI Agent GUI")

tk.Label(root, text="Enter your AI agent goal:").pack(pady=5)

entry = tk.Entry(root, width=80)
entry.pack(padx=10, pady=5)

run_button = tk.Button(root, text="Run Agent", command=analyze_goal)
run_button.pack(pady=5)

pdf_button = tk.Button(root, text="Generate PDF Report", command=generate_pdf_report)
pdf_button.pack(pady=5)

output_text = scrolledtext.ScrolledText(root, width=90, height=15)
output_text.pack(padx=10, pady=10)

chart_frame = tk.Frame(root)
chart_frame.pack(padx=10, pady=10)

root.mainloop()
