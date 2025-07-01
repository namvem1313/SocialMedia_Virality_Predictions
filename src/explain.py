import shap
import matplotlib.pyplot as plt
import os

def explain_model(model, X, output_dir="outputs"):
    """
    Generate SHAP summary plot and save to file.
    
    Args:
        model: Trained XGBoost model
        X: Feature DataFrame
        output_dir: Directory to save the plot
        
    Returns:
        shap_values: SHAP values object
    """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    output_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"✅ SHAP summary plot saved to {output_path}")

    return shap_values
