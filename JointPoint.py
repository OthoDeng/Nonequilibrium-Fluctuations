import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import os

class JointDistributionAnalyzer:
    def __init__(self, data_path="/Users/ottodeng/Desktop/Fluctuation/ERA5SLP/"):
        self.data_path = data_path
        self.variables = {}
        
    def load_data(self, var_names=['tp', 'w', 'tcrw']):
        """Load ERA5 data for specified variables"""
        for var in var_names:
            file_path = os.path.join(self.data_path, f"mean_{var}.nc")
            try:
                self.variables[var] = xr.open_dataset(file_path)
                print(f"Loaded {var} data from {file_path}")
            except FileNotFoundError:
                print(f"Warning: Could not find {file_path}")
    
    def prepare_data_for_analysis(self):
        """Flatten spatial data for joint analysis"""
        data_dict = {}
        
        for var_name, dataset in self.variables.items():
            # Get the main variable from dataset
            var_key = list(dataset.data_vars)[0]
            data = dataset[var_key]
            
            # Flatten spatial dimensions, keep time if present
            if 'time' in data.dims:
                # Average over time or select specific time
                data = data.mean(dim='time')
            
            # Flatten to 1D array
            data_dict[var_name] = data.values.flatten()
        
        # Remove NaN values
        df = pd.DataFrame(data_dict)
        df = df.dropna()
        
        return df
    
    def create_joint_plots(self, df, save_path="../ERA5SLP/fig7"):
        """Create joint distribution plots using seaborn"""
        os.makedirs(save_path, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 8)
        
        # Plot 1: Total Precipitation vs Vertical Motion
        if 'tp' in df.columns and 'w' in df.columns:
            fig1 = plt.figure(figsize=(12, 8))
            
            # Joint plot with hexbin
            g1 = sns.jointplot(data=df, x='w', y='tp', 
                              kind='hex', gridsize=30,
                              marginal_kws=dict(bins=50))
            g1.set_axis_labels('Vertical Motion (w)', 'Total Precipitation (tp)')
            plt.suptitle('Joint Distribution: Total Precipitation vs Vertical Motion', 
                        y=1.02, fontsize=14)
            plt.savefig(os.path.join(save_path, 'joint_tp_w_hex.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Joint plot with contour
            g2 = sns.jointplot(data=df, x='w', y='tp', 
                              kind='kde', fill=True)
            g2.set_axis_labels('Vertical Motion (w)', 'Total Precipitation (tp)')
            plt.suptitle('Joint Distribution: Total Precipitation vs Vertical Motion (Contour)', 
                        y=1.02, fontsize=14)
            plt.savefig(os.path.join(save_path, 'joint_tp_w_contour.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 2: Total Precipitation vs Total Column Water Vapor
        if 'tp' in df.columns and 'tcrw' in df.columns:
            g3 = sns.jointplot(data=df, x='tcrw', y='tp', 
                              kind='hex', gridsize=30,
                              marginal_kws=dict(bins=50))
            g3.set_axis_labels('Total Column Water Vapor (tcrw)', 'Total Precipitation (tp)')
            plt.suptitle('Joint Distribution: Total Precipitation vs Water Vapor', 
                        y=1.02, fontsize=14)
            plt.savefig(os.path.join(save_path, 'joint_tp_tcrw_hex.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            g4 = sns.jointplot(data=df, x='tcrw', y='tp', 
                              kind='kde', fill=True)
            g4.set_axis_labels('Total Column Water Vapor (tcrw)', 'Total Precipitation (tp)')
            plt.suptitle('Joint Distribution: Total Precipitation vs Water Vapor (Contour)', 
                        y=1.02, fontsize=14)
            plt.savefig(os.path.join(save_path, 'joint_tp_tcrw_contour.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_2d_histograms(self, df, save_path="../results/"):
        """Create 2D histograms for joint distributions"""
        os.makedirs(save_path, exist_ok=True)
        
        # 2D Histogram: TP vs W
        if 'tp' in df.columns and 'w' in df.columns:
            plt.figure(figsize=(10, 8))
            plt.hist2d(df['w'], df['tp'], bins=50, density=True, cmap='Blues')
            plt.colorbar(label='Density')
            plt.xlabel('Vertical Motion (w)')
            plt.ylabel('Total Precipitation (tp)')
            plt.title('2D Histogram: Total Precipitation vs Vertical Motion')
            plt.savefig(os.path.join(save_path, '2d_hist_tp_w.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2D Histogram: TP vs TCRW
        if 'tp' in df.columns and 'tcrw' in df.columns:
            plt.figure(figsize=(10, 8))
            plt.hist2d(df['tcrw'], df['tp'], bins=50, density=True, cmap='Reds')
            plt.colorbar(label='Density')
            plt.xlabel('Total Column Water Vapor (tcrw)')
            plt.ylabel('Total Precipitation (tp)')
            plt.title('2D Histogram: Total Precipitation vs Water Vapor')
            plt.savefig(os.path.join(save_path, '2d_hist_tp_tcrw.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def calculate_correlations(self, df):
        """Calculate correlation coefficients and mutual information"""
        results = {}
        
        # Pearson correlation
        if 'tp' in df.columns and 'w' in df.columns:
            pearson_tp_w, p_val_pearson_tp_w = pearsonr(df['tp'], df['w'])
            spearman_tp_w, p_val_spearman_tp_w = spearmanr(df['tp'], df['w'])
            mi_tp_w = mutual_info_regression(df[['w']], df['tp'])[0]
            
            results['tp_w'] = {
                'pearson': pearson_tp_w,
                'pearson_pval': p_val_pearson_tp_w,
                'spearman': spearman_tp_w,
                'spearman_pval': p_val_spearman_tp_w,
                'mutual_info': mi_tp_w
            }
        
        if 'tp' in df.columns and 'tcrw' in df.columns:
            pearson_tp_tcrw, p_val_pearson_tp_tcrw = pearsonr(df['tp'], df['tcrw'])
            spearman_tp_tcrw, p_val_spearman_tp_tcrw = spearmanr(df['tp'], df['tcrw'])
            mi_tp_tcrw = mutual_info_regression(df[['tcrw']], df['tp'])[0]
            
            results['tp_tcrw'] = {
                'pearson': pearson_tp_tcrw,
                'pearson_pval': p_val_pearson_tp_tcrw,
                'spearman': spearman_tp_tcrw,
                'spearman_pval': p_val_spearman_tp_tcrw,
                'mutual_info': mi_tp_tcrw
            }
        
        return results
    
    def print_correlation_results(self, results):
        """Print correlation results in a formatted way"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS RESULTS")
        print("="*60)
        
        for var_pair, metrics in results.items():
            print(f"\n{var_pair.upper().replace('_', ' vs ')}:")
            print(f"  Pearson correlation:  {metrics['pearson']:.4f} (p-value: {metrics['pearson_pval']:.4e})")
            print(f"  Spearman correlation: {metrics['spearman']:.4f} (p-value: {metrics['spearman_pval']:.4e})")
            print(f"  Mutual Information:   {metrics['mutual_info']:.4f}")
    
    def run_full_analysis(self):
        """Run complete joint distribution analysis"""
        print("Starting Joint Distribution Analysis...")
        
        # Load data
        self.load_data()
        
        # Prepare data
        df = self.prepare_data_for_analysis()
        print(f"Prepared data shape: {df.shape}")
        print(f"Available variables: {list(df.columns)}")
        
        # Create visualizations
        print("Creating joint plots...")
        self.create_joint_plots(df)
        
        print("Creating 2D histograms...")
        self.create_2d_histograms(df)
        
        # Calculate correlations
        print("Calculating correlations...")
        correlation_results = self.calculate_correlations(df)
        
        # Print results
        self.print_correlation_results(correlation_results)
        
        print(f"\nAnalysis complete! Check for visualizations.")
        
        return df, correlation_results

# Main execution
if __name__ == "__main__":
    analyzer = JointDistributionAnalyzer()
    df, results = analyzer.run_full_analysis()
