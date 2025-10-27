import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time


class NeuralNetwork:
    """Fully functional neural network with backpropagation"""
    
    def __init__(self, architecture, learning_rate=0.1):
        self.architecture = architecture
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        
        # Xavier initialization
        for i in range(len(architecture) - 1):
            scale = np.sqrt(2.0 / architecture[i])
            w = np.random.randn(architecture[i], architecture[i + 1]) * scale
            b = np.zeros(architecture[i + 1])
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        activations = [x]
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            activations.append(a)
        
        return activations
    
    def backward(self, x, y):
        activations = self.forward(x)
        output = activations[-1]
        
        # Calculate loss
        loss = np.mean((output - y) ** 2)
        
        # Backpropagation
        deltas = [(output - y) * self.sigmoid_derivative(output)]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * np.outer(activations[i], deltas[i])
            self.biases[i] -= self.lr * deltas[i]
        
        return output, loss
    
    def predict(self, x):
        activations = self.forward(x)
        return activations[-1]


class TetrisNeuralNetGUI:
    """Tetris-themed Neural Network GUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("NEURAL NET // VISUAL INTERFACE")
        self.root.configure(bg='black')
        
        # Network parameters
        self.architecture = [4, 8, 8, 3]
        self.learning_rate = 0.1
        self.network = NeuralNetwork(self.architecture, self.learning_rate)
        
        # Training state
        self.is_training = False
        self.epoch = 0
        self.loss = 1.0
        self.accuracy = 0.0
        self.history = {'loss': [], 'accuracy': []}
        self.animation_frame = 0
        
        self.setup_ui()
        self.create_training_data()
        self.start_animation()
    
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header with decorative blocks
        header = tk.Frame(main_frame, bg='white', bd=4, relief=tk.SOLID)
        header.pack(fill=tk.X, pady=(0, 10))
        
        header_content = tk.Frame(header, bg='white')
        header_content.pack(fill=tk.X, padx=10, pady=10)
        
        # Left decorative blocks
        left_blocks = tk.Frame(header_content, bg='white')
        left_blocks.pack(side=tk.LEFT)
        for i in range(3):
            block = tk.Frame(left_blocks, bg='black', width=15, height=15, bd=2, relief=tk.SOLID)
            block.pack(side=tk.LEFT, padx=2)
        
        title_label = tk.Label(
            header_content, 
            text="⚡ NEURAL NET // GUI ⚡",
            font=('Courier', 24, 'bold'),
            bg='white',
            fg='black'
        )
        title_label.pack(side=tk.LEFT, padx=20)
        
        # Right decorative blocks
        right_blocks = tk.Frame(header_content, bg='white')
        right_blocks.pack(side=tk.RIGHT)
        for i in range(3):
            block = tk.Frame(right_blocks, bg='black', width=15, height=15, bd=2, relief=tk.SOLID)
            block.pack(side=tk.LEFT, padx=2)
        
        # Content area
        content = tk.Frame(main_frame, bg='black')
        content.pack(fill=tk.BOTH, expand=True)
        
        
        left_frame = tk.Frame(content, bg='white', bd=4, relief=tk.SOLID)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        viz_header = tk.Label(
            left_frame,
            text="▼▼▼ NETWORK TOPOLOGY ▼▼▼",
            font=('Courier', 12, 'bold'),
            bg='black',
            fg='white',
            pady=8
        )
        viz_header.pack(fill=tk.X)
        
        # Network visualization canvas
        self.viz_figure = Figure(figsize=(7, 5), facecolor='black')
        self.viz_ax = self.viz_figure.add_subplot(111)
        self.viz_ax.set_facecolor('black')
        self.viz_canvas = FigureCanvasTkAgg(self.viz_figure, left_frame)
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Metrics display with Tetris blocks
        metrics_outer = tk.Frame(left_frame, bg='white', bd=2, relief=tk.SOLID)
        metrics_outer.pack(fill=tk.X, padx=2, pady=2)
        
        metrics_frame = tk.Frame(metrics_outer, bg='black')
        metrics_frame.pack(fill=tk.X, pady=8)
        
        # Epoch metric
        epoch_box = tk.Frame(metrics_frame, bg='white', bd=3, relief=tk.SOLID)
        epoch_box.pack(side=tk.LEFT, padx=15, pady=5)
        epoch_inner = tk.Frame(epoch_box, bg='black')
        epoch_inner.pack(padx=3, pady=3)
        tk.Label(epoch_inner, text="EPOCH", font=('Courier', 9), bg='black', fg='white').pack()
        self.epoch_label = tk.Label(
            epoch_inner,
            text="0",
            font=('Courier', 18, 'bold'),
            bg='black',
            fg='white'
        )
        self.epoch_label.pack()
        
        # Loss metric
        loss_box = tk.Frame(metrics_frame, bg='white', bd=3, relief=tk.SOLID)
        loss_box.pack(side=tk.LEFT, padx=15, pady=5)
        loss_inner = tk.Frame(loss_box, bg='black')
        loss_inner.pack(padx=3, pady=3)
        tk.Label(loss_inner, text="LOSS", font=('Courier', 9), bg='black', fg='white').pack()
        self.loss_label = tk.Label(
            loss_inner,
            text="1.0000",
            font=('Courier', 18, 'bold'),
            bg='black',
            fg='white'
        )
        self.loss_label.pack()
        
        # Accuracy metric
        acc_box = tk.Frame(metrics_frame, bg='white', bd=3, relief=tk.SOLID)
        acc_box.pack(side=tk.LEFT, padx=15, pady=5)
        acc_inner = tk.Frame(acc_box, bg='black')
        acc_inner.pack(padx=3, pady=3)
        tk.Label(acc_inner, text="ACCURACY", font=('Courier', 9), bg='black', fg='white').pack()
        self.acc_label = tk.Label(
            acc_inner,
            text="0.0%",
            font=('Courier', 18, 'bold'),
            bg='black',
            fg='white'
        )
        self.acc_label.pack()
        
        # Right side - Controls
        right_frame = tk.Frame(content, bg='black')
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configuration panel
        config_frame = tk.Frame(right_frame, bg='white', bd=4, relief=tk.SOLID)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        config_header = tk.Label(
            config_frame,
            text="⚙⚙⚙ CONFIGURATION ⚙⚙⚙",
            font=('Courier', 11, 'bold'),
            bg='black',
            fg='white',
            pady=6
        )
        config_header.pack(fill=tk.X)
        
        config_content = tk.Frame(config_frame, bg='white')
        config_content.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            config_content,
            text="Architecture:",
            font=('Courier', 10, 'bold'),
            bg='white'
        ).pack(anchor=tk.W)
        
        arch_box = tk.Frame(config_content, bg='black', bd=2, relief=tk.SOLID)
        arch_box.pack(fill=tk.X, pady=(2, 10))
        
        self.arch_entry = tk.Entry(
            arch_box,
            font=('Courier', 11),
            bg='black',
            fg='white',
            insertbackground='white',
            bd=0
        )
        self.arch_entry.insert(0, '-'.join(map(str, self.architecture)))
        self.arch_entry.pack(padx=2, pady=2)
        
        tk.Label(
            config_content,
            text="Learning Rate:",
            font=('Courier', 10, 'bold'),
            bg='white'
        ).pack(anchor=tk.W)
        
        lr_box = tk.Frame(config_content, bg='black', bd=2, relief=tk.SOLID)
        lr_box.pack(fill=tk.X, pady=(2, 0))
        
        self.lr_entry = tk.Entry(
            lr_box,
            font=('Courier', 11),
            bg='black',
            fg='white',
            insertbackground='white',
            bd=0
        )
        self.lr_entry.insert(0, str(self.learning_rate))
        self.lr_entry.pack(padx=2, pady=2)
        
        # Control buttons
        control_frame = tk.Frame(right_frame, bg='white', bd=4, relief=tk.SOLID)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        control_header = tk.Label(
            control_frame,
            text="▶▶▶ CONTROLS ▶▶▶",
            font=('Courier', 11, 'bold'),
            bg='black',
            fg='white',
            pady=6
        )
        control_header.pack(fill=tk.X)
        
        button_frame = tk.Frame(control_frame, bg='white')
        button_frame.pack(padx=10, pady=10)
        
        self.train_btn = tk.Button(
            button_frame,
            text="▶ START TRAINING",
            font=('Courier', 11, 'bold'),
            bg='white',
            fg='black',
            activebackground='black',
            activeforeground='white',
            command=self.start_training,
            bd=4,
            relief=tk.SOLID,
            cursor='hand2'
        )
        self.train_btn.pack(fill=tk.X, pady=5, ipady=5)
        
        self.stop_btn = tk.Button(
            button_frame,
            text="■ STOP",
            font=('Courier', 11, 'bold'),
            bg='black',
            fg='white',
            activebackground='white',
            activeforeground='black',
            command=self.stop_training,
            bd=4,
            relief=tk.SOLID,
            state=tk.DISABLED,
            cursor='hand2'
        )
        self.stop_btn.pack(fill=tk.X, pady=5, ipady=5)
        
        self.reset_btn = tk.Button(
            button_frame,
            text="↻ RESET NETWORK",
            font=('Courier', 11, 'bold'),
            bg='black',
            fg='white',
            activebackground='white',
            activeforeground='black',
            command=self.reset_network,
            bd=4,
            relief=tk.SOLID,
            cursor='hand2'
        )
        self.reset_btn.pack(fill=tk.X, pady=5, ipady=5)
        
        # Predictions panel
        pred_frame = tk.Frame(right_frame, bg='white', bd=4, relief=tk.SOLID)
        pred_frame.pack(fill=tk.BOTH, expand=True)
        
        pred_header = tk.Label(
            pred_frame,
            text="◆◆◆ PREDICTIONS ◆◆◆",
            font=('Courier', 11, 'bold'),
            bg='black',
            fg='white',
            pady=6
        )
        pred_header.pack(fill=tk.X)
        
        pred_box = tk.Frame(pred_frame, bg='black', bd=2, relief=tk.SOLID)
        pred_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.pred_text = tk.Text(
            pred_box,
            font=('Courier', 9),
            bg='black',
            fg='white',
            height=15,
            width=30,
            bd=0
        )
        self.pred_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.draw_network()
    
    def create_training_data(self):
        """Create pattern classification dataset"""
        self.train_x = np.array([
            [0, 0, 0, 0],  # Class 0: all zeros
            [1, 1, 1, 1],  # Class 1: all ones
            [1, 0, 1, 0],  # Class 2: alternating
            [0, 1, 0, 1],  # Class 2: alternating
            [1, 1, 0, 0],  # Class 1: mostly ones
            [0, 0, 1, 1],  # Class 1: mostly ones
            [1, 0, 0, 0],  # Class 0: mostly zeros
            [0, 1, 0, 0],  # Class 0: mostly zeros
            [0, 0, 0, 1],  # Class 0: mostly zeros
            [1, 0, 0, 1],  # Class 2: edges
            [0, 1, 1, 0],  # Class 2: middle
        ])
        
        self.train_y = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
        ])
    
    def draw_network(self):
        """Draw neural network topology with ALL parallel connections"""
        self.viz_ax.clear()
        self.viz_ax.set_xlim(0, 10)
        self.viz_ax.set_ylim(0, 10)
        self.viz_ax.axis('off')
        
        layers = self.architecture
        layer_x = np.linspace(1.5, 8.5, len(layers))
        
        # Calculate all node positions
        node_positions = []
        for i, n_nodes in enumerate(layers):
            y_positions = np.linspace(1.5, 8.5, n_nodes)
            positions = [(layer_x[i], y) for y in y_positions]
            node_positions.append(positions)
        
        # Draw ALL connections between layers
        for layer_idx in range(len(layers) - 1):
            current_layer = node_positions[layer_idx]
            next_layer = node_positions[layer_idx + 1]
            
            # Draw every connection from current to next layer
            for i, (x1, y1) in enumerate(current_layer):
                for j, (x2, y2) in enumerate(next_layer):
                    # Get weight if network exists
                    if self.network and layer_idx < len(self.network.weights):
                        weight = self.network.weights[layer_idx][i][j]
                        alpha = min(0.6, abs(weight) * 0.4 + 0.1)
                        linewidth = max(0.3, abs(weight) * 1.5)
                    else:
                        alpha = 0.15
                        linewidth = 0.5
                    
                    # Animate during training
                    if self.is_training:
                        pulse = np.sin(self.animation_frame * 0.1 + i * 0.3 + j * 0.2)
                        alpha = alpha * (0.8 + pulse * 0.2)
                    
                    self.viz_ax.plot(
                        [x1, x2],
                        [y1, y2],
                        'w-',
                        alpha=alpha,
                        linewidth=linewidth,
                        solid_capstyle='round'
                    )
        
        # Draw nodes as Tetris blocks
        for layer_idx, positions in enumerate(node_positions):
            for node_idx, (x, y) in enumerate(positions):
                size = 0.25
                
                # Pulse effect during training
                if self.is_training:
                    pulse = np.sin(self.animation_frame * 0.15 + node_idx * 0.5) * 0.03
                    size += pulse
                
                # Outer white square
                outer = plt.Rectangle(
                    (x - size, y - size),
                    size * 2, size * 2,
                    facecolor='white',
                    edgecolor='white',
                    linewidth=2
                )
                self.viz_ax.add_patch(outer)
                
                # Inner black square
                inner_size = size * 0.65
                inner = plt.Rectangle(
                    (x - inner_size, y - inner_size),
                    inner_size * 2, inner_size * 2,
                    facecolor='black',
                    edgecolor='black'
                )
                self.viz_ax.add_patch(inner)
                
                # Active indicator for output layer during training
                if self.is_training and layer_idx == len(node_positions) - 1:
                    glow_size = inner_size * 0.5
                    glow_alpha = 0.6 + np.sin(self.animation_frame * 0.2 + node_idx) * 0.3
                    glow = plt.Rectangle(
                        (x - glow_size, y - glow_size),
                        glow_size * 2, glow_size * 2,
                        facecolor='white',
                        alpha=glow_alpha
                    )
                    self.viz_ax.add_patch(glow)
        
        self.viz_canvas.draw()
    
    def start_animation(self):
        """Animate the network visualization"""
        self.animation_frame += 1
        if self.is_training or self.animation_frame % 3 == 0:
            self.draw_network()
        self.root.after(50, self.start_animation)
    
    def start_training(self):
        """Start training in a separate thread"""
        if not self.is_training:
            self.is_training = True
            self.train_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.arch_entry.config(state=tk.DISABLED)
            self.lr_entry.config(state=tk.DISABLED)
            
            training_thread = threading.Thread(target=self.train_loop)
            training_thread.daemon = True
            training_thread.start()
    
    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.train_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.arch_entry.config(state=tk.NORMAL)
        self.lr_entry.config(state=tk.NORMAL)
    
    def train_loop(self):
        """Training loop"""
        self.history = {'loss': [], 'accuracy': []}
        
        for epoch in range(200):
            if not self.is_training:
                break
            
            # Shuffle data
            indices = np.random.permutation(len(self.train_x))
            
            total_loss = 0
            correct = 0
            
            for idx in indices:
                output, loss = self.network.backward(self.train_x[idx], self.train_y[idx])
                total_loss += loss
                
                pred_class = np.argmax(output)
                true_class = np.argmax(self.train_y[idx])
                
                if pred_class == true_class:
                    correct += 1
            
            self.epoch = epoch + 1
            self.loss = total_loss / len(self.train_x)
            self.accuracy = (correct / len(self.train_x)) * 100
            
            self.history['loss'].append(self.loss)
            self.history['accuracy'].append(self.accuracy)
            
            # Update UI
            self.root.after(0, self.update_metrics)
            
            time.sleep(0.03)
        
        # Generate predictions after training
        self.root.after(0, self.generate_predictions)
        self.stop_training()
    
    def update_metrics(self):
        """Update metric labels"""
        self.epoch_label.config(text=f"{self.epoch}")
        self.loss_label.config(text=f"{self.loss:.4f}")
        self.acc_label.config(text=f"{self.accuracy:.1f}%")
    
    def generate_predictions(self):
        """Generate and display predictions"""
        test_inputs = [
            [0.1, 0.2, 0.1, 0.0],
            [0.9, 0.8, 0.9, 1.0],
            [0.8, 0.1, 0.9, 0.2],
            [0.0, 0.9, 0.1, 0.8],
        ]
        
        self.pred_text.delete(1.0, tk.END)
        self.pred_text.insert(tk.END, "═" * 32 + "\n")
        self.pred_text.insert(tk.END, "   TEST PREDICTIONS\n")
        self.pred_text.insert(tk.END, "═" * 32 + "\n\n")
        
        for i, inp in enumerate(test_inputs):
            output = self.network.predict(np.array(inp))
            pred_class = np.argmax(output)
            confidence = output[pred_class] * 100
            
            self.pred_text.insert(tk.END, f"╔═ Input {i+1} ═══════════════╗\n")
            self.pred_text.insert(tk.END, f"║ {inp}\n")
            self.pred_text.insert(tk.END, f"║\n")
            self.pred_text.insert(tk.END, f"║ → CLASS {pred_class}\n")
            self.pred_text.insert(tk.END, f"║   ({confidence:.1f}% confident)\n")
            self.pred_text.insert(tk.END, f"║\n")
            
            for j, val in enumerate(output):
                bar_length = int(val * 18)
                bar = "█" * bar_length + "░" * (18 - bar_length)
                self.pred_text.insert(tk.END, f"║ C{j} {bar} {val:.3f}\n")
            
            self.pred_text.insert(tk.END, f"╚═══════════════════════════╝\n\n")
    
    def reset_network(self):
        """Reset the network"""
        try:
            arch_str = self.arch_entry.get()
            self.architecture = [int(x) for x in arch_str.split('-')]
            self.learning_rate = float(self.lr_entry.get())
            
            self.network = NeuralNetwork(self.architecture, self.learning_rate)
            self.epoch = 0
            self.loss = 1.0
            self.accuracy = 0.0
            self.history = {'loss': [], 'accuracy': []}
            
            self.update_metrics()
            self.draw_network()
            self.pred_text.delete(1.0, tk.END)
            self.pred_text.insert(tk.END, "\n\n")
            self.pred_text.insert(tk.END, "  ╔═══════════════════════╗\n")
            self.pred_text.insert(tk.END, "  ║   NETWORK RESET!      ║\n")
            self.pred_text.insert(tk.END, "  ║                       ║\n")
            self.pred_text.insert(tk.END, "  ║   Ready to train...   ║\n")
            self.pred_text.insert(tk.END, "  ╚═══════════════════════╝\n")
            
        except Exception as e:
            print(f"Error resetting network: {e}")


def main():
    root = tk.Tk()
    root.geometry("1100x750")
    app = TetrisNeuralNetGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()