import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.database import DatabaseConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CancellationModel:
    """ML Model untuk Prediksi Cancelation Rate"""
    
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.db_connector = DatabaseConnector(db_config)
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = None
    
    def validate_database(self) -> bool:
        """Validate database dan schema"""
        logger.info("=== DATABASE VALIDATION ===")
        
        # Connect
        if not self.db_connector.connect():
            logger.error("Failed to connect to database")
            return False
        
        # Check tables exist
        required_tables = ['fact_reservation', 'dim_status']
        for table in required_tables:
            if not self.db_connector.check_table_exists(table):
                logger.error(f" Table '{table}' not found")
                return False
            else:
                stats = self.db_connector.get_table_stats(table)
                logger.info(f" {table}: {stats['row_count']} rows, {stats['size_mb']} MB")
        
        # Check critical columns
        critical_columns = {
            'fact_reservation': [
                'lead_time_days', 'length_of_stay_days', 'room_nights',
                'total_revenue', 'revenue_per_night', 'room_rate',
                'adults', 'children', 'other_guests', 'qty_rooms', 'fk_status'
            ],
            'dim_status': ['pk_status', 'status_name']
        }
        
        for table, columns in critical_columns.items():
            missing_cols = []
            for col in columns:
                if not self.db_connector.check_column_exists(table, col):
                    missing_cols.append(col)
            
            if missing_cols:
                logger.error(f" {table} missing columns: {missing_cols}")
                return False
            else:
                logger.info(f" {table} has all required columns")
        
        logger.info(" Database validation passed\n")
        return True
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """Load training data dengan error handling"""
        logger.info(" LOADING DATA ")
        
        # Safe query 
        query = """
        SELECT 
            fr.lead_time_days,
            fr.length_of_stay_days,
            fr.room_nights,
            fr.total_revenue,
            fr.revenue_per_night,
            fr.room_rate,
            fr.adults,
            fr.children,
            fr.other_guests,
            fr.qty_rooms,
            COALESCE(dstat.status_name, 'Unknown') as status_name,
            fr.pk_fact_id
        FROM fact_reservation fr
        LEFT JOIN dim_status dstat ON fr.fk_status = dstat.pk_status
        WHERE fr.lead_time_days IS NOT NULL
          AND fr.length_of_stay_days IS NOT NULL
          AND fr.total_revenue IS NOT NULL
          AND fr.total_revenue > 0
        """
        
        try:
            df = self.db_connector.execute_query(query, fetch=True)
            
            if df is None or len(df) == 0:
                logger.error(" No data returned from query")
                return None
            
            logger.info(f" Loaded {len(df)} records")
            logger.info(f" Status distribution:\n{df['status_name'].value_counts()}\n")
            
            return df
            
        except Exception as e:
            logger.error(f" Error loading data: {e}")
            return None
    
    def prepare_features(self, df: pd.DataFrame) -> Optional[tuple]:
        """Prepare features dengan validation"""
        logger.info("=== PREPARING FEATURES ===")
        
        try:
            # Target: 1 = Cancelled, 0 = Active
            y = (df['status_name'] == 'Cancelled').astype(int)
            
            # Features
            feature_cols = [
                'lead_time_days', 'length_of_stay_days', 'room_nights',
                'total_revenue', 'revenue_per_night', 'room_rate',
                'adults', 'children', 'other_guests', 'qty_rooms'
            ]
            
            X = df[feature_cols].fillna(0).copy()
            
            # Check for missing columns
            missing_cols = [col for col in feature_cols if col not in X.columns]
            if missing_cols:
                logger.error(f" Missing columns: {missing_cols}")
                return None
            
            self.feature_names = feature_cols
            
            logger.info(f"Features shape: {X.shape}")
            logger.info(f"Features: {feature_cols}")
            logger.info(f"Target distribution:\n{y.value_counts()}")
            logger.info(f"Class balance: {y.value_counts(normalize=True).to_dict()}\n")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    def train(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train models """
        logger.info("=== TRAINING MODELS ===")
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Models
            models_config = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            }
            results = []
            best_model_name = None
            best_roc_auc = 0
            
            for name, model in models_config.items():
                logger.info(f"\n Training {name}...")
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                metrics = {
                    'Model': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                    'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
                }
                
                results.append(metrics)
                self.models[name] = model
                
                # Log results
                logger.info(f" {name}:")
                for k, v in metrics.items():
                    if k != 'Model':
                        logger.info(f"  {k:15s}: {v:.4f}")
                
                # Log confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                logger.info(f"  Confusion Matrix:\n    TN={cm[0,0]}, FP={cm[0,1]}")
                logger.info(f"    FN={cm[1,0]}, TP={cm[1,1]}")
                
                # Track best model
                if metrics['ROC-AUC'] > best_roc_auc:
                    best_roc_auc = metrics['ROC-AUC']
                    best_model_name = name
            
            # Select best model
            logger.info(f"\n Best Model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
            self.models['best'] = self.models[best_model_name]
            
            # Feature importance (dari best model)
            if hasattr(self.models['best'], 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': self.models['best'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                logger.info(f"\n Top 5 Important Features:")
                for idx, row in self.feature_importance.head(5).iterrows():
                    logger.info(f"  {row['feature']:20s}: {row['importance']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f" Training error: {e}")
            return False
    
    def save_model(self, path: str = 'models'):
        """Save model & artifacts"""
        try:
            Path(path).mkdir(exist_ok=True)
            
            joblib.dump(self.models['best'], f"{path}/model.pkl")
            joblib.dump(self.scaler, f"{path}/scaler.pkl")
            joblib.dump(self.feature_names, f"{path}/features.pkl")
            
            if self.feature_importance is not None:
                self.feature_importance.to_csv(f"{path}/feature_importance.csv", index=False)
            
            logger.info(f" Model saved to {path}/")
            
        except Exception as e:
            logger.error(f" Save error: {e}")
if __name__ == '__main__':
    # Database config
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'samsara_db',
        'user': 'postgres',
        'password': ''  
    }
    
    # Train pipeline
    trainer = CancellationModel(db_config)
    if not trainer.validate_database():
        logger.error(" Validation failed. Exiting.")
        exit(1)
    df = trainer.load_data()
    if df is None:
        logger.error(" Data loading failed. Exiting.")
        exit(1)
    result = trainer.prepare_features(df)
    if result is None:
        logger.error(" Feature preparation failed. Exiting.")
        exit(1)
    X, y = result
    if not trainer.train(X, y):
        logger.error(" Training failed. Exiting.")
        exit(1)

    trainer.save_model('models')
    
    trainer.db_connector.close()
    
    logger.info("\n Pipeline completed successfully!")