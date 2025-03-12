import gradio as gr
from src.inference import predict

def main():
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Sketchpad(width=560, height=560, brush=gr.Brush(default_size=25)),
        outputs=gr.Label(num_top_classes=3),
        title="LeNet Handwritten Digit Classifier",
        description="Draw a digit and press 'Submit' to classify it.",
        theme="dark"
    )
    interface.launch(share=True)

if __name__ == '__main__':
    main()
