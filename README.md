# Eye Tracking and Mathematical Comparison Analysis Framework

## Project Overview

This is a comprehensive analysis framework for studying mathematical comparison tasks using behavioral and eye tracking data. The system provides tools to analyze how different participant groups perform comparison tasks with various numerical formats, examining performance metrics, visual attention patterns, and reaction times across different experimental conditions.

## Experimental Framework

### Supported Experimental Designs

**Participant Group Classifications:**
- **Group A (Theo)**: Participants with theoretical/analytical background
- **Group B (Thet)**: Participants with applied/practical background

## Analysis Capabilities

### Performance Metrics Analyzed

**Behavioral Measures:**
- Reaction time analysis with outlier detection
- Accuracy calculations across conditions
- Response pattern classification
- Participant performance profiling

**Statistical Methods:**
- Mixed-effects modeling for repeated measures
- Between-group and within-group comparisons
- Multiple comparison corrections
- Effect size calculations

### Eye Tracking Features

**Attention Analysis:**
The framework supports comprehensive eye tracking analysis including:

**Areas of Interest (AOI) Support:**
- Equation/stimulus areas
- Multiple choice options (A, B, C, etc.)
- Custom definable regions
- Automatic fixation classification

**Attention Metrics:**
- Fixation duration and frequency
- Attention distribution patterns
- Gaze transition analysis
- Time-based attention shifts

**Analysis Capabilities:**
- Statistical comparison between participant groups
- Condition-based attention differences
- Multi-level statistical modeling
- Visualization of attention patterns

### Requirements

```bash
pip install -r requirements.txt
```
## Usage

### Running Main Analysis

```bash
# Run Experiment 1 analysis with median method
python main.py --exp_num 1 --method median

# Run Experiment 2 analysis with mean method  
python main.py --exp_num 2 --method mean

# Run Experiment 2b analysis
python main.py --exp_num 2b --method median
```

### Running Eye Tracking Analysis

```bash
python eye_tracker_analysis.py
```

### Preprocessing Data

```bash
# Preprocess individual experiments
python preprocess_exp1.py
python preprocess_exp2.py

# Preprocess eye tracking data
python preprocess_eyetracker.py
```

## Analysis Methods

### Statistical Tests Performed

1. **Generalized Estimating Equations (GEE)**
   - Logistic regression for accuracy data
   - Linear regression for reaction time data
   - Accounts for repeated measures structure

2. **Outlier Detection & Filtering**
   - Median Absolute Deviation (MAD) method
   - Configurable outlier bounds (default: 3 MAD)

3. **Post-hoc Comparisons**
   - Bonferroni correction for multiple comparisons
   - Pairwise contrasts between all item types

4. **N-Way ANOVA**
   - Analysis of variance for eye tracking data
   - LSD post-hoc testing when significant effects found

5. **Kruskal-Wallis Tests**
   - Non-parametric alternative for non-normal data
   - Post-hoc Dunn's test for pairwise comparisons

### Output Files

**For each experiment:**
- `*_accuracy.png`: Accuracy comparison visualizations
- `*_reaction_time_*.png`: Reaction time analysis plots  
- `*_outlier_*.png`: Outlier detection visualizations
- Statistical summaries and detailed results (customizable output formats)

## Key Features

### Data Processing
- Automatic outlier detection and filtering
- Flexible analysis methods (mean vs median)
- Robust handling of missing data
- Multiple choice accuracy calculations for Experiment 2b

### Statistical Analysis
- Mixed-effects modeling with GEE
- Comprehensive pairwise comparisons
- Multiple testing corrections
- Effect size calculations

### Visualization
- Interactive plotting with matplotlib/seaborn
- Automatic chart generation and saving
- Customizable plot parameters
- Professional statistical graphics

### Eye Tracking Integration
- Area of Interest (AOI) classification
- Fixation pattern analysis
- Attention distribution statistics
- Multi-level statistical modeling

## Example Use Cases

This framework can be applied to various research scenarios:

- **Educational Research**: Comparing learning approaches across different academic backgrounds
- **Cognitive Psychology**: Studying numerical cognition and mathematical processing
- **User Experience**: Analyzing visual attention in interface design
- **Methodological Research**: Developing analysis pipelines for eye tracking studies

## Disclaimer

This repository contains analysis code and methodological frameworks developed for research purposes. The code is provided as-is for educational and research use. Any included visualizations or data files serve purely as examples to demonstrate the analytical capabilities and output formats of the framework. These examples do not represent findings from any specific published research.

## License

MIT License - This code is provided for educational and research purposes. Feel free to use, modify, and distribute according to the MIT License terms.
