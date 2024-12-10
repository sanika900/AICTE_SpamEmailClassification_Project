import streamlit as st
import pickle

# Load the pre-trained model and vectorizer
try:
    model = pickle.load(open('spam123.pkl', 'rb'))
    vectorizer = pickle.load(open('vec123.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

def main():
    st.title("📧 Email Spam Classification App")
    st.write("Use this application to classify whether an email is spam or not.")
    st.markdown("### Features:")
    st.markdown("- Input an email and classify it.")
    st.markdown("- Uses Machine Learning for classification.")
    
    # Input section
    st.subheader("Classification")
    user_email = st.text_area("Enter the email content here:", height=150)
    
    # Button for classification
    if st.button("Classify Email"):
        if user_email.strip():
            try:
                # Transform and classify
                email_data = [user_email]
                email_vector = vectorizer.transform(email_data).toarray()
                prediction = model.predict(email_vector)
                
                # Display result
                if prediction[0] == 0:
                    st.success("✅ This is NOT a Spam Email.")
                else:
                    st.error("❌ This is a Spam Email.")
            except Exception as e:
                st.error(f"An error occurred during classification: {e}")
        else:
            st.warning("Please enter some email content before classifying.")
    
    # Reset input field
    if st.button("Reset"):
        st.experimental_rerun()

if __name__ == '__main__':
    main()
