# ============================================================
# AI PROTOTYPE: CUSTOMER SUPPORT TICKET CLASSIFICATION
# ============================================================

# Install first:
# pip install transformers torch

from transformers import pipeline


# ------------------------------------------------------------
# 1. Load AI Model
# ------------------------------------------------------------

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


# ------------------------------------------------------------
# 2. Define Categories
# ------------------------------------------------------------

categories = [
    "Billing",
    "Technical Issue",
    "Account Access",
    "General Inquiry"
]


# ------------------------------------------------------------
# 3. Classification Function
# ------------------------------------------------------------

def classify_ticket(message):

    result = classifier(
        message,
        candidate_labels=categories
    )

    predicted_category = result["labels"][0]
    confidence = result["scores"][0]

    return predicted_category, confidence


# ------------------------------------------------------------
# 4. Example Inputs
# ------------------------------------------------------------

example_tickets = [
    "Why was I charged twice for my subscription?",
    
    "The application crashes whenever I upload a file.",
    
    "I forgot my password and cannot log into my account.",
    
    "What are your customer support hours?"
]


# ------------------------------------------------------------
# 5. Generate Predictions
# ------------------------------------------------------------

print("=" * 60)
print("CUSTOMER SUPPORT AI CLASSIFICATION PROTOTYPE")
print("=" * 60)

for ticket in example_tickets:

    category, confidence = classify_ticket(ticket)

    print("\nInput:")
    print(ticket)

    print("Output:")
    print("Category:", category)
    print("Confidence:", f"{confidence:.2%}")


# ------------------------------------------------------------
# 6. Interactive Mode
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("INTERACTIVE MODE")
print("=" * 60)

while True:

    message = input(
        "\nEnter a customer support message "
        "(type 'exit' to stop): "
    )

    if message.lower() == "exit":
        print("Prototype terminated.")
        break

    category, confidence = classify_ticket(message)

    print("\nAI Output")
    print("-" * 30)
    print("Predicted Category:", category)
    print("Confidence:", f"{confidence:.2%}")
    
    # ------------------------------------------------------------
    # 7. Frontend Interface
    # ------------------------------------------------------------
    
demo = gr.Interface(
    fn=classify_ticket,
    inputs=gr.Textbox(
        label="Customer Support Message",
        placeholder="Example: I forgot my password..."
    ),
    outputs=gr.Textbox(
        label="AI Classification"
    ),
    title="AI Customer Support Ticket Classifier",
    description=(
        "Enter a customer-support message and "
        "the AI will classify it into one of four categories."
    )
)

demo.launch()