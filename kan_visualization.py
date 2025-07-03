"""
Standalone visualization script for KAN training curves
Run this after training to see the comparison plots
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to path so we can import KAN_Network
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from kan_network import KAN_Network
except ImportError:
    print("Error: Could not import KAN_Network. Make sure you're in the correct directory.")
    sys.exit(1)

def run_training_comparison():
    """Run training comparison and create visualizations"""
    
    print("="*80)
    print("KAN Network Training Visualization")
    print("="*80)
    
    # Target function: f(x) = sin(3*pi*x) * exp(-x^2)
    def target_function_1d(x):
        return torch.sin(3 * torch.pi * x) * torch.exp(-x**2)
    
    # Generate training and test data
    torch.manual_seed(123)
    x_train = torch.linspace(-2, 2, 400).unsqueeze(1)
    y_train = target_function_1d(x_train)
    
    # Test data extends beyond training range
    x_test = torch.linspace(-2.5, 2.5, 100).unsqueeze(1)
    y_test = target_function_1d(x_test)
    
    print(f"Target function: f(x) = sin(3πx) * exp(-x²)")
    print(f"Training range: [{x_train.min():.1f}, {x_train.max():.1f}]")
    print(f"Test range: [{x_test.min():.1f}, {x_test.max():.1f}]")
    
    # Create networks
    kan_static = KAN_Network([1, 8, 8, 1], num=5, k=3)
    kan_adaptive = KAN_Network([1, 8, 8, 1], num=5, k=3)
    kan_adaptive.load_state_dict(kan_static.state_dict())
    
    # Optimizers
    optimizer_static = optim.Adam(kan_static.parameters(), lr=0.01)
    optimizer_adaptive = optim.Adam(kan_adaptive.parameters(), lr=0.01)
    
    # Training parameters
    epochs = 200
    grid_update_frequency = 50
    
    print(f"\nTraining both networks for {epochs} epochs...")
    print(f"Adaptive network: grid updates every {grid_update_frequency} epochs")
    
    # Storage for losses
    train_losses_static = []
    train_losses_adaptive = []
    test_losses_static = []
    test_losses_adaptive = []
    
    # Training loop
    for epoch in range(epochs):
        # Train static network
        optimizer_static.zero_grad()
        y_pred_static = kan_static(x_train)
        train_loss_static = F.mse_loss(y_pred_static, y_train)
        train_loss_static.backward()
        optimizer_static.step()
        
        # Train adaptive network
        optimizer_adaptive.zero_grad()
        y_pred_adaptive = kan_adaptive(x_train)
        train_loss_adaptive = F.mse_loss(y_pred_adaptive, y_train)
        train_loss_adaptive.backward()
        optimizer_adaptive.step()
        
        # Record training losses
        train_losses_static.append(train_loss_static.item())
        train_losses_adaptive.append(train_loss_adaptive.item())
        
        # Evaluate on test set
        with torch.no_grad():
            test_pred_static = kan_static(x_test)
            test_pred_adaptive = kan_adaptive(x_test)
            test_loss_static = F.mse_loss(test_pred_static, y_test)
            test_loss_adaptive = F.mse_loss(test_pred_adaptive, y_test)
            test_losses_static.append(test_loss_static.item())
            test_losses_adaptive.append(test_loss_adaptive.item())
        
        # Grid updates for adaptive network
        if (epoch + 1) % grid_update_frequency == 0:
            print(f"  Epoch {epoch+1}: Updating grid... (Train: {train_loss_adaptive:.6f}, Test: {test_loss_adaptive:.6f})")
            kan_adaptive.update_grid_from_samples(x_train, mode='sample')
        
        # Progress updates
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: Static [T:{train_loss_static:.6f}, V:{test_loss_static:.6f}] | Adaptive [T:{train_loss_adaptive:.6f}, V:{test_loss_adaptive:.6f}]")
    
    # Create comprehensive visualization
    create_training_plots(
        epochs, train_losses_static, train_losses_adaptive,
        test_losses_static, test_losses_adaptive, grid_update_frequency
    )
    
    # Analysis
    print_analysis(train_losses_static, train_losses_adaptive, 
                  test_losses_static, test_losses_adaptive)

def create_training_plots(epochs, train_static, train_adaptive, test_static, test_adaptive, update_freq):
    """Create comprehensive training visualization plots"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs_range = range(1, epochs + 1)
    
    # Plot 1: Training Loss Comparison
    ax1 = axes[0, 0]
    ax1.plot(epochs_range, train_static, 'b-', label='Static Grid', linewidth=2, alpha=0.8)
    ax1.plot(epochs_range, train_adaptive, 'r-', label='Adaptive Grid', linewidth=2, alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (log scale)')
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark grid updates
    for epoch in range(update_freq, epochs + 1, update_freq):
        ax1.axvline(x=epoch, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Plot 2: Test Loss Comparison  
    ax2 = axes[0, 1]
    ax2.plot(epochs_range, test_static, 'b-', label='Static Grid', linewidth=2, alpha=0.8)
    ax2.plot(epochs_range, test_adaptive, 'r-', label='Adaptive Grid', linewidth=2, alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Loss (log scale)')
    ax2.set_title('Test Loss Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Mark grid updates
    for epoch in range(update_freq, epochs + 1, update_freq):
        ax2.axvline(x=epoch, color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    # Plot 3: Static Network Train vs Test
    ax3 = axes[1, 0]
    ax3.plot(epochs_range, train_static, 'g-', label='Training', linewidth=2, alpha=0.8)
    ax3.plot(epochs_range, test_static, 'orange', label='Test', linewidth=2, alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (log scale)')
    ax3.set_title('Static Network: Train vs Test', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Adaptive Network Train vs Test
    ax4 = axes[1, 1]
    ax4.plot(epochs_range, train_adaptive, 'g-', label='Training', linewidth=2, alpha=0.8)
    ax4.plot(epochs_range, test_adaptive, 'orange', label='Test', linewidth=2, alpha=0.8)
    ax4.set_yscale('log')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss (log scale)')
    ax4.set_title('Adaptive Network: Train vs Test', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Mark grid updates and highlight overfitting
    for epoch in range(update_freq, epochs + 1, update_freq):
        ax4.axvline(x=epoch, color='red', linestyle='--', alpha=0.6, linewidth=1,
                   label='Grid Update' if epoch == update_freq else '')
    
    # Find and mark minimum test loss point
    min_test_epoch = np.argmin(test_adaptive) + 1
    min_test_loss = min(test_adaptive)
    ax4.plot(min_test_epoch, min_test_loss, 'ro', markersize=8, label=f'Min Test (Epoch {min_test_epoch})')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('kan_training_curves_detailed.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved as 'kan_training_curves_detailed.png'")
    plt.show()

def print_analysis(train_static, train_adaptive, test_static, test_adaptive):
    """Print detailed analysis of training results"""
    
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    # Final performance
    print(f"\nFinal Performance:")
    print(f"  📊 Training Loss:")
    print(f"     Static:   {train_static[-1]:.8f}")
    print(f"     Adaptive: {train_adaptive[-1]:.8f}")
    train_improvement = ((train_static[-1] - train_adaptive[-1]) / train_static[-1] * 100)
    print(f"     Improvement: {train_improvement:+.2f}%")
    
    print(f"\n  📊 Test Loss:")
    print(f"     Static:   {test_static[-1]:.8f}")
    print(f"     Adaptive: {test_adaptive[-1]:.8f}")
    test_improvement = ((test_static[-1] - test_adaptive[-1]) / test_static[-1] * 100)
    print(f"     Change: {test_improvement:+.2f}%")
    
    # Overfitting analysis
    print(f"\n🔍 Overfitting Analysis:")
    
    # Best test performance for each network
    best_test_static_idx = np.argmin(test_static)
    best_test_adaptive_idx = np.argmin(test_adaptive)
    
    print(f"  Static Network:")
    print(f"     Best test at epoch: {best_test_static_idx + 1}")
    print(f"     Best test loss: {test_static[best_test_static_idx]:.8f}")
    print(f"     Final test loss: {test_static[-1]:.8f}")
    static_degradation = ((test_static[-1] / test_static[best_test_static_idx]) - 1) * 100
    print(f"     Test degradation: {static_degradation:+.2f}%")
    
    print(f"  Adaptive Network:")
    print(f"     Best test at epoch: {best_test_adaptive_idx + 1}")
    print(f"     Best test loss: {test_adaptive[best_test_adaptive_idx]:.8f}")
    print(f"     Final test loss: {test_adaptive[-1]:.8f}")
    adaptive_degradation = ((test_adaptive[-1] / test_adaptive[best_test_adaptive_idx]) - 1) * 100
    print(f"     Test degradation: {adaptive_degradation:+.2f}%")
    
    # Generalization gap
    print(f"\n📏 Generalization Gap (Test/Train ratio):")
    static_gap = test_static[-1] / train_static[-1]
    adaptive_gap = test_adaptive[-1] / train_adaptive[-1]
    print(f"  Static Network: {static_gap:.2f}")
    print(f"  Adaptive Network: {adaptive_gap:.2f}")
    
    if adaptive_gap < static_gap:
        print(f"  ✅ Adaptive has smaller generalization gap")
    else:
        print(f"  ❌ Adaptive has larger generalization gap (overfitting)")
    
    # Summary
    print(f"\n🎯 SUMMARY:")
    if train_improvement > 0 and test_improvement > 0:
        print(f"  ✅ Adaptive grid refinement: CLEAR WIN")
        print(f"     Better training AND better test performance")
    elif train_improvement > 0 and test_improvement < 0:
        print(f"  ⚠️  Adaptive grid refinement: OVERFITTING")
        print(f"     Better training but worse test performance")
    elif train_improvement < 0 and test_improvement > 0:
        print(f"  🤔 Adaptive grid refinement: REGULARIZATION EFFECT")
        print(f"     Worse training but better test performance")
    else:
        print(f"  ❌ Adaptive grid refinement: NO BENEFIT")
        print(f"     Worse on both training and test")

if __name__ == "__main__":
    try:
        run_training_comparison()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc() 