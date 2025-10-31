import os
import pandas as pd
import altair as alt
import argparse

## plotting settings
if True:  # In order to bypass isort when saving
    from altairThemes import altairThemes
alt.themes.register("publishTheme", altairThemes.publishTheme)
alt.themes.enable("publishTheme")

def plot_loss_curve(losses, save_path):
    """
    Plot the loss curve from a list of losses and save it as an HTML file.
    
    Parameters:
    - losses: List of loss values.
    - save_path: Path to save the HTML file.
    """
    df_L = []
    for each in losses:
        df_L.append(pd.read_csv(each))
    df = pd.concat(df_L, ignore_index=True)
    df = df.groupby('epoch').mean().reset_index()
    df = df[['epoch', 'train_gen_loss', 'val_gen_loss']]
    df = df.melt(id_vars='epoch', var_name='loss_type', value_name='loss_value')
    df.rename(columns={'loss_type': 'loss_type', 'loss_value': 'loss'}, inplace=True)
    chart = alt.Chart(df).mark_line().encode(
        x='epoch',
        y='loss:Q',
        color='loss_type:N',
    )
    chart.interactive().save(os.path.join(save_path, "loss_curve.html"))
    print(f"Loss curve saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot loss curve from CSV files.")
    parser.add_argument("--losses", type=str, nargs='+', required=True, help="List of paths to CSV files containing losses.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the HTML file with the loss curve.")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    plot_loss_curve(args.losses, args.save_path)

if __name__ == "__main__":
    main()

