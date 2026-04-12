#!/bin/bash
# Common experiment patterns for the training pipeline
# Usage: bash run_experiments.sh <experiment_type>

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="$SCRIPT_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Experiment 1: Compare BatchNorm Effect
exp_batchnorm_comparison() {
    print_header "EXPERIMENT 1: BatchNorm Effect (CIFAR-10, CNN)"
    
    print_success "Training WITHOUT BatchNorm..."
    python src/training/train.py \
        --dataset cifar10 \
        --model-type cnn \
        --use-bn False \
        --num-epochs 10 \
        --seed 42 \
        --save-dir "$RESULTS_DIR/exp1"
    
    print_success "Training WITH BatchNorm..."
    python src/training/train.py \
        --dataset cifar10 \
        --model-type cnn \
        --use-bn True \
        --num-epochs 10 \
        --seed 42 \
        --save-dir "$RESULTS_DIR/exp1"
    
    print_success "Generating comparison report..."
    python src/training/analyze_results.py \
        --results-dir "$RESULTS_DIR/exp1" \
        --plot \
        --report
}

# Experiment 2: Architecture Comparison
exp_architecture_comparison() {
    print_header "EXPERIMENT 2: MLP vs CNN (Fashion-MNIST, with BatchNorm)"
    
    print_success "Training MLP..."
    python src/training/train.py \
        --dataset fashion_mnist \
        --model-type mlp \
        --use-bn True \
        --num-epochs 10 \
        --seed 42 \
        --save-dir "$RESULTS_DIR/exp2"
    
    print_success "Training CNN..."
    python src/training/train.py \
        --dataset fashion_mnist \
        --model-type cnn \
        --use-bn True \
        --num-epochs 10 \
        --seed 42 \
        --save-dir "$RESULTS_DIR/exp2"
    
    print_success "Generating comparison report..."
    python src/training/analyze_results.py \
        --results-dir "$RESULTS_DIR/exp2" \
        --plot \
        --report
}

# Experiment 3: Learning Rate Sweep
exp_learning_rate_sweep() {
    print_header "EXPERIMENT 3: Learning Rate Sweep (CIFAR-10, CNN, with BatchNorm)"
    
    for lr in 0.0001 0.0005 0.001 0.005 0.01; do
        print_success "Training with learning rate $lr..."
        python src/training/train.py \
            --dataset cifar10 \
            --model-type cnn \
            --use-bn True \
            --learning-rate $lr \
            --num-epochs 10 \
            --seed 42 \
            --save-dir "$RESULTS_DIR/exp3"
    done
    
    print_success "Generating comparison report..."
    python src/training/analyze_results.py \
        --results-dir "$RESULTS_DIR/exp3" \
        --plot \
        --report
}

# Experiment 4: Different Seeds (for stability analysis)
exp_multiple_seeds() {
    print_header "EXPERIMENT 4: Multiple Seeds Stability (CIFAR-10, CNN, with BatchNorm)"
    
    for seed in 42 123 456 789 999; do
        print_success "Training with seed $seed..."
        python src/training/train.py \
            --dataset cifar10 \
            --model-type cnn \
            --use-bn True \
            --num-epochs 10 \
            --seed $seed \
            --save-dir "$RESULTS_DIR/exp4"
    done
    
    print_success "Generating stability report..."
    python src/training/analyze_results.py \
        --results-dir "$RESULTS_DIR/exp4" \
        --plot \
        --report
}

# Experiment 5: Comprehensive Comparison
exp_comprehensive() {
    print_header "EXPERIMENT 5: Comprehensive Comparison (All Combinations)"
    
    datasets=("cifar10" "fashion_mnist")
    models=("mlp" "cnn")
    bn_options=("True" "False")
    
    total=$((${#datasets[@]} * ${#models[@]} * ${#bn_options[@]}))
    current=0
    
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            for use_bn in "${bn_options[@]}"; do
                current=$((current + 1))
                print_success "[$current/$total] Training $dataset/$model/bn=$use_bn..."
                
                python src/training/train.py \
                    --dataset $dataset \
                    --model-type $model \
                    --use-bn $use_bn \
                    --num-epochs 10 \
                    --seed 42 \
                    --save-dir "$RESULTS_DIR/exp5"
            done
        done
    done
    
    print_success "Generating comprehensive report..."
    python src/training/analyze_results.py \
        --results-dir "$RESULTS_DIR/exp5" \
        --plot \
        --report
}

# Experiment 6: Quick Test (for debugging)
exp_quick_test() {
    print_header "EXPERIMENT 6: Quick Test (Small training for debugging)"
    
    print_success "Quick training run..."
    python src/training/train.py \
        --dataset cifar10 \
        --model-type cnn \
        --use-bn True \
        --num-epochs 2 \
        --batch-size 128 \
        --seed 42 \
        --save-dir "$RESULTS_DIR/exp6"
    
    print_success "Quick analysis..."
    python src/training/analyze_results.py \
        --results-dir "$RESULTS_DIR/exp6"
}

# Show help
show_help() {
    cat << EOF
Usage: bash run_experiments.sh <experiment_type>

Available experiments:
  1. batchnorm      - Compare with/without BatchNorm on CIFAR-10 CNN
  2. architecture   - Compare MLP vs CNN on Fashion-MNIST
  3. learning_rate  - Learning rate sweep on CIFAR-10 CNN
  4. multiple_seeds - Train with different seeds for stability analysis
  5. comprehensive  - All combinations (datasets × models × bn)
  6. quick_test     - Quick test for debugging

Examples:
  bash run_experiments.sh batchnorm
  bash run_experiments.sh architecture
  bash run_experiments.sh learning_rate

Results will be saved to: $RESULTS_DIR/
EOF
}

# Main
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    case "$1" in
        1|batchnorm)
            exp_batchnorm_comparison
            ;;
        2|architecture)
            exp_architecture_comparison
            ;;
        3|learning_rate)
            exp_learning_rate_sweep
            ;;
        4|multiple_seeds)
            exp_multiple_seeds
            ;;
        5|comprehensive)
            exp_comprehensive
            ;;
        6|quick_test)
            exp_quick_test
            ;;
        *)
            echo "Unknown experiment: $1"
            show_help
            exit 1
            ;;
    esac
    
    print_header "Experiment completed! Results saved to $RESULTS_DIR"
}

main "$@"
