import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sub_models.attention_blocks import get_per_head_fixed_mask_causal, get_fixed_mask_causal


def visualize_attention_mask_heatmap(mask, title="Attention Mask", head_idx=None, figsize=(12, 10)):
    """
    Visualize a single attention mask as a heatmap.
    
    Args:
        mask: Attention mask tensor of shape (batch, seq_len, seq_len) or (batch, heads, seq_len, seq_len)
        title: Title for the plot
        head_idx: If mask has multiple heads, specify which head to visualize
        figsize: Figure size tuple
    """
    # Convert mask to numpy if it's a tensor
    if torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.array(mask)
    
    # Extract the mask data
    if mask_np.ndim == 4:  # (batch, heads, seq_len, seq_len)
        if head_idx is not None:
            mask_data = mask_np[0, head_idx]
        else:
            raise ValueError("For multi-head masks, please specify head_idx")
    elif mask_np.ndim == 3:  # (batch, seq_len, seq_len)
        mask_data = mask_np[0]
    elif mask_np.ndim == 2:  # (seq_len, seq_len)
        mask_data = mask_np
    else:
        raise ValueError(f"Expected mask dimension 2, 3 or 4, got {mask_np.ndim}")
    
    # Convert -inf to a very negative value for visualization
    mask_vis = np.where(mask_data == float('-inf'), -10, mask_data)
    mask_vis = np.where(np.isneginf(mask_data), -10, mask_vis)
    
    plt.figure(figsize=figsize)
    
    # Create custom colormap: red for masked (-inf), green for allowed (0), blue for penalties
    colors = ['darkred', 'red', 'yellow', 'lightgreen', 'green']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('mask_cmap', colors, N=n_bins)
    
    # Create heatmap
    im = plt.imshow(mask_vis, cmap=cmap, aspect='equal', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Mask Value', rotation=270, labelpad=20, fontsize=12)
    
    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Key Position', fontsize=14)
    plt.ylabel('Query Position', fontsize=14)
    
    # Add grid for better readability
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, mask_vis.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mask_vis.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
    
    # Add text annotations for small matrices
    if mask_vis.shape[0] <= 16:
        for i in range(mask_vis.shape[0]):
            for j in range(mask_vis.shape[1]):
                if mask_vis[i, j] == -10:  # -inf
                    text = '∞'
                    color = 'white'
                elif mask_vis[i, j] == 0:
                    text = '0'
                    color = 'black'
                else:
                    text = f'{mask_vis[i, j]:.1f}'
                    color = 'black'
                plt.text(j, i, text, ha='center', va='center', 
                        fontsize=8, color=color, fontweight='bold')
    
    plt.tight_layout()
    return plt.gcf()


def visualize_multi_head_masks(batch_length, mask_percents, flag=False, soft=True, 
                              soft_penalty=-1.0, device='cpu', figsize=(20, 15)):
    """
    Visualize multiple attention head masks in a grid layout.
    
    Args:
        batch_length: Sequence length
        mask_percents: List of masking percentages for each head
        flag: Whether to use random masking
        soft: Whether to use soft masking
        soft_penalty: Penalty value for soft masking
        device: Device to create tensors on
        figsize: Figure size tuple
    """
    # Generate the multi-head mask
    mask = get_per_head_fixed_mask_causal(
        batch_length, mask_percents, flag, soft, device, soft_penalty
    )
    
    num_heads = len(mask_percents)
    
    # Calculate grid dimensions
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Ensure axes is always a 2D array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    else:
        # axes is already 2D when rows > 1 and cols > 1
        pass
    
    # Create custom colormap
    colors = ['darkred', 'red', 'yellow', 'lightgreen', 'green']
    cmap = LinearSegmentedColormap.from_list('mask_cmap', colors, N=100)
    
    for h in range(num_heads):
        row = h // cols
        col = h % cols
        ax = axes[row, col]
        
        # Extract mask for this head
        if torch.is_tensor(mask):
            mask_data = mask[0, h].detach().cpu().numpy()
        else:
            mask_data = np.array(mask[0, h])
        mask_vis = np.where(mask_data == float('-inf'), -10, mask_data)
        mask_vis = np.where(np.isneginf(mask_data), -10, mask_vis)
        
        # Create heatmap
        im = ax.imshow(mask_vis, cmap=cmap, aspect='equal', interpolation='nearest')
        
        # Customize subplot
        ax.set_title(f'Head {h+1} (Mask: {mask_percents[h]*100:.1f}%)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, batch_length, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, batch_length, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
        
        # Add text annotations for small matrices
        if batch_length <= 12:
            for i in range(batch_length):
                for j in range(batch_length):
                    if mask_vis[i, j] == -10:  # -inf
                        text = '∞'
                        color = 'white'
                    elif mask_vis[i, j] == 0:
                        text = '0'
                        color = 'black'
                    else:
                        text = f'{mask_vis[i, j]:.1f}'
                        color = 'black'
                    ax.text(j, i, text, ha='center', va='center', 
                           fontsize=7, color=color, fontweight='bold')
    
    # Hide empty subplots
    for h in range(num_heads, rows * cols):
        row = h // cols
        col = h % cols
        ax = axes[row, col]
        ax.set_visible(False)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label('Mask Value (-∞: Blocked, 0: Allowed, <0: Penalized)', 
                   rotation=270, labelpad=20, fontsize=12)
    
    # Main title
    mask_type = "Random" if flag else "Sequential"
    penalty_type = "Soft" if soft else "Hard"
    fig.suptitle(f'{penalty_type} {mask_type} Masking - Multi-Head Attention Patterns\n'
                f'Sequence Length: {batch_length}, Heads: {num_heads}', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig


def compare_masking_strategies(batch_length=16, mask_percents=[0.0, 0.25, 0.5, 0.75], 
                             device='cpu', figsize=(20, 12)):
    """
    Compare different masking strategies: sequential vs random, hard vs soft.
    
    Args:
        batch_length: Sequence length
        mask_percents: List of masking percentages
        device: Device to create tensors on
        figsize: Figure size tuple
    """
    strategies = [
        ('Sequential Hard', False, False, -1.0),
        ('Sequential Soft', False, True, -1.0),
        ('Random Hard', True, False, -1.0),
        ('Random Soft', True, True, -1.0)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Create custom colormap
    colors = ['darkred', 'red', 'yellow', 'lightgreen', 'green']
    cmap = LinearSegmentedColormap.from_list('mask_cmap', colors, N=100)
    
    for idx, (strategy_name, flag, soft, soft_penalty) in enumerate(strategies):
        # Generate mask for one head with 50% masking
        mask = get_per_head_fixed_mask_causal(
            batch_length, [0.5], flag, soft, device, soft_penalty
        )
        
        if torch.is_tensor(mask):
            mask_data = mask[0, 0].detach().cpu().numpy()
        else:
            mask_data = np.array(mask[0, 0])
        mask_vis = np.where(mask_data == float('-inf'), -10, mask_data)
        mask_vis = np.where(np.isneginf(mask_data), -10, mask_vis)
        
        ax = axes[idx]
        im = ax.imshow(mask_vis, cmap=cmap, aspect='equal', interpolation='nearest')
        
        ax.set_title(f'{strategy_name} (50% Masking)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, batch_length, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, batch_length, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
        
        # Add text annotations for small matrices
        if batch_length <= 12:
            for i in range(batch_length):
                for j in range(batch_length):
                    if mask_vis[i, j] == -10:  # -inf
                        text = '∞'
                        color = 'white'
                    elif mask_vis[i, j] == 0:
                        text = '0'
                        color = 'black'
                    else:
                        text = f'{mask_vis[i, j]:.1f}'
                        color = 'black'
                    ax.text(j, i, text, ha='center', va='center', 
                           fontsize=8, color=color, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label('Mask Value (-∞: Blocked, 0: Allowed, <0: Penalized)', 
                   rotation=270, labelpad=20, fontsize=12)
    
    fig.suptitle(f'Comparison of Masking Strategies\nSequence Length: {batch_length}', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    return fig


def visualize_mask_percentage_effects(batch_length=16, num_heads=4, device='cpu', figsize=(20, 12)):
    """
    Show how different mask percentages affect attention patterns.
    
    Args:
        batch_length: Sequence length
        num_heads: Number of heads to show
        device: Device to create tensors on
        figsize: Figure size tuple
    """
    mask_percentages = [0.0, 0.25, 0.5, 0.75]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Create custom colormap
    colors = ['darkred', 'red', 'yellow', 'lightgreen', 'green']
    cmap = LinearSegmentedColormap.from_list('mask_cmap', colors, N=100)
    
    for idx, mask_percent in enumerate(mask_percentages):
        # Generate mask for one head
        mask = get_per_head_fixed_mask_causal(
            batch_length, [mask_percent], False, True, device, -1.0
        )
        
        if torch.is_tensor(mask):
            mask_data = mask[0, 0].detach().cpu().numpy()
        else:
            mask_data = np.array(mask[0, 0])
        mask_vis = np.where(mask_data == float('-inf'), -10, mask_data)
        mask_vis = np.where(np.isneginf(mask_data), -10, mask_vis)
        
        ax = axes[idx]
        im = ax.imshow(mask_vis, cmap=cmap, aspect='equal', interpolation='nearest')
        
        ax.set_title(f'{mask_percent*100:.0f}% Masking', fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, batch_length, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, batch_length, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
        
        # Calculate and display statistics
        total_positions = np.sum(np.tril(np.ones((batch_length, batch_length))))
        masked_positions = np.sum(mask_vis == -10)
        actual_mask_percent = masked_positions / total_positions * 100
        
        ax.text(0.02, 0.98, f'Actual: {actual_mask_percent:.1f}% masked', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label('Mask Value (-∞: Blocked, 0: Allowed, <0: Penalized)', 
                   rotation=270, labelpad=20, fontsize=12)
    
    fig.suptitle(f'Effect of Mask Percentage on Attention Patterns\nSequence Length: {batch_length}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    return fig


def create_comprehensive_mask_analysis(batch_length=16, device='cpu'):
    """
    Create a comprehensive analysis showing various aspects of the masking system.
    """
    print("Creating comprehensive mask analysis...")
    
    # Example 1: Multi-head with different percentages
    print("1. Multi-head attention with diversified masking...")
    mask_percents = [0.0, 0.25, 0.5, 0.75]
    fig1 = visualize_multi_head_masks(
        batch_length, mask_percents, flag=False, soft=True, 
        soft_penalty=-1.0, device=device
    )
    fig1.savefig('multi_head_masks.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example 2: Compare masking strategies
    print("2. Comparing masking strategies...")
    fig2 = compare_masking_strategies(batch_length, device=device)
    fig2.savefig('masking_strategies_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example 3: Effect of mask percentages
    print("3. Effect of different mask percentages...")
    fig3 = visualize_mask_percentage_effects(batch_length, device=device)
    fig3.savefig('mask_percentage_effects.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Example 4: Specialized heads configuration
    print("4. Specialized heads configuration...")
    specialized_percents = [0.0, 0.1, 0.3, 0.6, 0.8, 0.9]
    fig4 = visualize_multi_head_masks(
        batch_length, specialized_percents, flag=False, soft=True, 
        soft_penalty=-2.0, device=device, figsize=(24, 12)
    )
    fig4.savefig('specialized_heads_masks.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analysis complete! Images saved as PNG files.")
    
    return fig1, fig2, fig3, fig4


def quick_demo(batch_length=8, save_plots=False):
    """
    Quick demonstration of mask visualization capabilities.
    
    Args:
        batch_length: Sequence length for demonstration
        save_plots: Whether to save plots as PNG files
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Quick Mask Visualization Demo (device: {device})")
    print("=" * 50)
    
    try:
        # Demo 1: Basic multi-head visualization
        print("1. Multi-head attention with different masking percentages...")
        mask_percents = [0.0, 0.3, 0.6, 0.9]
        fig1 = visualize_multi_head_masks(
            batch_length, mask_percents, flag=False, soft=True, 
            soft_penalty=-1.0, device=device, figsize=(14, 8)
        )
        if save_plots:
            fig1.savefig('demo_multi_head.png', dpi=200, bbox_inches='tight')
            print("   Saved: demo_multi_head.png")
        else:
            plt.show()
        
        # Demo 2: Strategy comparison
        print("\n2. Comparing masking strategies...")
        fig2 = compare_masking_strategies(batch_length, device=device, figsize=(12, 8))
        if save_plots:
            fig2.savefig('demo_strategies.png', dpi=200, bbox_inches='tight')
            print("   Saved: demo_strategies.png")
        else:
            plt.show()
        
        print("\n✓ Demo completed successfully!")
        
        if not save_plots:
            print("\nTo save plots, call: quick_demo(save_plots=True)")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run quick demo by default
    quick_demo(batch_length=10, save_plots=True)
