# KMRL AI-Driven Train Induction Planning System - Complete Backend
# Requirements: flask, pandas, numpy, scikit-learn, pulp, joblib, sqlalchemy
#
# Install: pip install flask pandas numpy scikit-learn pulp joblib sqlalchemy flask-cors
# Run: python backend.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import pickle
import logging

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Optimization
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("Warning: PuLP not installed. Using heuristic solver only.")

# Setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== DATA MODELS ====================

class TrainData:
    """Model for train configuration and state"""
    def __init__(self, train_id: str):
        if not train_id or len(train_id.strip()) == 0:
            raise ValueError("TrainID cannot be empty")
        self.train_id = train_id.strip().upper()
        self.fitness_rolling = "Valid"
        self.fitness_signal = "Valid"
        self.fitness_telecom = "Valid"
        self.fitness_rolling_expiry = None
        self.fitness_signal_expiry = None
        self.fitness_telecom_expiry = None
        self.job_card_open = False
        self.job_card_critical = False
        self.job_card_ids = []
        self.mileage_km = 0
        self.branding_contract = "None"
        self.branding_hours_current = 0
        self.branding_hours_required = 0
        self.cleaning_hours_required = 2
        self.home_bay = "B1"
        self.current_bay = "B1"
        self.last_service_date = None
        self.component_health_score = 100.0
        self.iot_sensor_alerts = []
        self.historical_performance = []
        
    def to_dict(self):
        return {
            "train_id": self.train_id,
            "fitness_rolling": self.fitness_rolling,
            "fitness_signal": self.fitness_signal,
            "fitness_telecom": self.fitness_telecom,
            "fitness_rolling_expiry": self.fitness_rolling_expiry,
            "fitness_signal_expiry": self.fitness_signal_expiry,
            "fitness_telecom_expiry": self.fitness_telecom_expiry,
            "job_card_open": self.job_card_open,
            "job_card_critical": self.job_card_critical,
            "job_card_ids": self.job_card_ids,
            "mileage_km": self.mileage_km,
            "branding_contract": self.branding_contract,
            "branding_hours_current": self.branding_hours_current,
            "branding_hours_required": self.branding_hours_required,
            "cleaning_hours_required": self.cleaning_hours_required,
            "home_bay": self.home_bay,
            "current_bay": self.current_bay,
            "last_service_date": self.last_service_date,
            "component_health_score": self.component_health_score,
            "iot_sensor_alerts": self.iot_sensor_alerts
        }

# ==================== DATA STORE ====================

class TrainRegistry:
    """Centralized registry for all trains with uniqueness enforcement"""
    def __init__(self):
        self.trains: Dict[str, TrainData] = {}
        self._load_from_disk()
    
    def add_train(self, train: TrainData) -> bool:
        """Add train, enforcing unique TrainID"""
        if train.train_id in self.trains:
            logger.warning(f"TrainID {train.train_id} already exists")
            return False
        self.trains[train.train_id] = train
        self._save_to_disk()
        logger.info(f"Added train {train.train_id}")
        return True
    
    def update_train(self, train_id: str, data: dict) -> bool:
        """Update existing train"""
        if train_id not in self.trains:
            return False
        train = self.trains[train_id]
        for key, value in data.items():
            if hasattr(train, key):
                setattr(train, key, value)
        self._save_to_disk()
        return True
    
    def get_train(self, train_id: str) -> Optional[TrainData]:
        return self.trains.get(train_id)
    
    def get_all_trains(self) -> List[TrainData]:
        return list(self.trains.values())
    
    def remove_train(self, train_id: str) -> bool:
        if train_id in self.trains:
            del self.trains[train_id]
            self._save_to_disk()
            return True
        return False
    
    def train_exists(self, train_id: str) -> bool:
        return train_id in self.trains
    
    def _save_to_disk(self):
        """Persist registry to disk"""
        try:
            data = {tid: train.to_dict() for tid, train in self.trains.items()}
            with open('train_registry.json', 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _load_from_disk(self):
        """Load registry from disk"""
        try:
            if os.path.exists('train_registry.json'):
                with open('train_registry.json', 'r') as f:
                    data = json.load(f)
                    for tid, train_dict in data.items():
                        train = TrainData(tid)
                        for key, value in train_dict.items():
                            if hasattr(train, key):
                                setattr(train, key, value)
                        self.trains[tid] = train
                logger.info(f"Loaded {len(self.trains)} trains from disk")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

# Global registry
registry = TrainRegistry()

# ==================== RULE ENGINE ====================

class RuleEngine:
    """Business rules and constraint checking"""
    
    @staticmethod
    def check_fitness_validity(train: TrainData, current_date: datetime) -> Tuple[bool, List[str]]:
        """Check all fitness certificates"""
        reasons = []
        is_valid = True
        
        if train.fitness_rolling.lower() == "expired":
            reasons.append("Rolling stock fitness expired")
            is_valid = False
        elif train.fitness_rolling_expiry:
            try:
                expiry = datetime.fromisoformat(train.fitness_rolling_expiry)
                if expiry < current_date:
                    reasons.append("Rolling stock fitness expired")
                    is_valid = False
            except:
                pass
        
        if train.fitness_signal.lower() == "expired":
            reasons.append("Signalling fitness expired")
            is_valid = False
        elif train.fitness_signal_expiry:
            try:
                expiry = datetime.fromisoformat(train.fitness_signal_expiry)
                if expiry < current_date:
                    reasons.append("Signalling fitness expired")
                    is_valid = False
            except:
                pass
        
        if train.fitness_telecom.lower() == "expired":
            reasons.append("Telecom fitness expired")
            is_valid = False
        elif train.fitness_telecom_expiry:
            try:
                expiry = datetime.fromisoformat(train.fitness_telecom_expiry)
                if expiry < current_date:
                    reasons.append("Telecom fitness expired")
                    is_valid = False
            except:
                pass
        
        return is_valid, reasons
    
    @staticmethod
    def check_job_card_status(train: TrainData) -> Tuple[bool, List[str]]:
        """Check maintenance job cards"""
        reasons = []
        is_valid = True
        
        if train.job_card_open:
            reasons.append("Open job card exists")
            if train.job_card_critical:
                reasons.append("CRITICAL job card - must be resolved")
                is_valid = False
            else:
                # Non-critical job cards don't disqualify but lower priority
                pass
        
        return is_valid, reasons
    
    @staticmethod
    def check_component_health(train: TrainData, threshold: float = 70.0) -> Tuple[bool, List[str]]:
        """Check overall component health"""
        reasons = []
        is_valid = True
        
        if train.component_health_score < threshold:
            reasons.append(f"Component health below threshold ({train.component_health_score:.1f}%)")
            if train.component_health_score < 50:
                is_valid = False
        
        if train.iot_sensor_alerts:
            reasons.append(f"{len(train.iot_sensor_alerts)} IoT sensor alerts")
        
        return is_valid, reasons
    
    @staticmethod
    def is_train_service_ready(train: TrainData, current_date: datetime) -> Tuple[bool, List[str]]:
        """Comprehensive service readiness check"""
        all_reasons = []
        
        fitness_valid, fitness_reasons = RuleEngine.check_fitness_validity(train, current_date)
        jobcard_valid, jobcard_reasons = RuleEngine.check_job_card_status(train)
        health_valid, health_reasons = RuleEngine.check_component_health(train)
        
        all_reasons.extend(fitness_reasons)
        all_reasons.extend(jobcard_reasons)
        all_reasons.extend(health_reasons)
        
        is_ready = fitness_valid and jobcard_valid and health_valid
        
        return is_ready, all_reasons

# ==================== SCORING ENGINE ====================

class ScoringEngine:
    """Multi-objective scoring for train prioritization"""
    
    def __init__(self, weights: Optional[Dict] = None):
        self.weights = weights or {
            "mileage": 0.30,
            "branding": 0.25,
            "cleaning": 0.15,
            "health": 0.20,
            "reliability": 0.10
        }
    
    def compute_mileage_score(self, train: TrainData, fleet_avg_mileage: float) -> float:
        """Lower mileage = higher priority (to balance wear)"""
        if fleet_avg_mileage == 0:
            return 0.5
        ratio = train.mileage_km / max(fleet_avg_mileage, 1)
        # Inverse sigmoid: prefer trains below average mileage
        return 1 / (1 + np.exp(5 * (ratio - 1)))
    
    def compute_branding_score(self, train: TrainData) -> float:
        """Higher shortfall = higher priority"""
        shortfall = max(0, train.branding_hours_required - train.branding_hours_current)
        if train.branding_hours_required == 0:
            return 0
        return shortfall / train.branding_hours_required
    
    def compute_cleaning_score(self, train: TrainData, max_cleaning_hours: int) -> float:
        """Lower cleaning requirement = higher score (less resource intensive)"""
        if max_cleaning_hours == 0:
            return 1.0
        return 1 - (train.cleaning_hours_required / max(max_cleaning_hours, 1))
    
    def compute_health_score(self, train: TrainData) -> float:
        """Direct mapping of component health"""
        return train.component_health_score / 100.0
    
    def compute_reliability_score(self, train: TrainData) -> float:
        """Based on historical performance"""
        if not train.historical_performance:
            return 0.8  # Neutral default
        # Calculate success rate from historical data
        successes = sum(1 for perf in train.historical_performance if perf.get("success", True))
        return successes / len(train.historical_performance)
    
    def compute_composite_score(self, train: TrainData, context: Dict) -> float:
        """Weighted composite score"""
        fleet_avg_mileage = context.get("fleet_avg_mileage", 0)
        max_cleaning_hours = context.get("max_cleaning_hours", 8)
        
        mileage_score = self.compute_mileage_score(train, fleet_avg_mileage)
        branding_score = self.compute_branding_score(train)
        cleaning_score = self.compute_cleaning_score(train, max_cleaning_hours)
        health_score = self.compute_health_score(train)
        reliability_score = self.compute_reliability_score(train)
        
        composite = (
            self.weights["mileage"] * mileage_score +
            self.weights["branding"] * branding_score +
            self.weights["cleaning"] * cleaning_score +
            self.weights["health"] * health_score +
            self.weights["reliability"] * reliability_score
        )
        
        return composite
    
    def score_all_trains(self, trains: List[TrainData], context: Dict) -> pd.DataFrame:
        """Score all trains and return sorted DataFrame"""
        scores = []
        for train in trains:
            composite = self.compute_composite_score(train, context)
            scores.append({
                "train_id": train.train_id,
                "composite_score": composite,
                "mileage_km": train.mileage_km,
                "branding_shortfall": max(0, train.branding_hours_required - train.branding_hours_current),
                "cleaning_hours": train.cleaning_hours_required,
                "health_score": train.component_health_score,
                "home_bay": train.home_bay
            })
        
        df = pd.DataFrame(scores)
        return df.sort_values("composite_score", ascending=False).reset_index(drop=True)

# ==================== OPTIMIZATION ENGINE ====================

class OptimizationEngine:
    """MIP-based optimization with constraints"""
    
    def __init__(self, use_pulp: bool = PULP_AVAILABLE):
        self.use_pulp = use_pulp
    
    def optimize(self, trains: List[TrainData], constraints: Dict, 
                 scored_df: pd.DataFrame, current_date: datetime) -> Dict:
        """
        Run optimization to assign trains to Service/Standby/IBL
        """
        if self.use_pulp:
            result = self._optimize_pulp(trains, constraints, scored_df, current_date)
            if result:
                return result
        
        # Fallback heuristic
        return self._optimize_heuristic(trains, constraints, scored_df, current_date)
    
    def _optimize_pulp(self, trains: List[TrainData], constraints: Dict,
                       scored_df: pd.DataFrame, current_date: datetime) -> Optional[Dict]:
        """PuLP MIP optimization"""
        try:
            import pulp
            
            prob = pulp.LpProblem("KMRL_Induction", pulp.LpMaximize)
            
            # Decision variables
            train_ids = [t.train_id for t in trains]
            x_service = pulp.LpVariable.dicts("service", train_ids, 0, 1, cat="Binary")
            x_standby = pulp.LpVariable.dicts("standby", train_ids, 0, 1, cat="Binary")
            x_ibl = pulp.LpVariable.dicts("ibl", train_ids, 0, 1, cat="Binary")
            
            # Score mapping
            score_map = dict(zip(scored_df["train_id"], scored_df["composite_score"]))
            
            # Objective: Maximize total service score
            prob += pulp.lpSum([score_map.get(tid, 0) * x_service[tid] for tid in train_ids])
            
            # Constraints
            for train in trains:
                tid = train.train_id
                # Each train assigned to exactly one state
                prob += x_service[tid] + x_standby[tid] + x_ibl[tid] == 1
                
                # Service readiness check
                is_ready, _ = RuleEngine.is_train_service_ready(train, current_date)
                if not is_ready:
                    prob += x_service[tid] == 0
                    prob += x_ibl[tid] == 1
            
            # Resource constraints
            max_service = constraints.get("max_service_trains", 20)
            prob += pulp.lpSum([x_service[tid] for tid in train_ids]) <= max_service
            
            cleaning_capacity = constraints.get("cleaning_capacity_hours", 40)
            cleaning_map = {t.train_id: t.cleaning_hours_required for t in trains}
            prob += pulp.lpSum([x_service[tid] * cleaning_map[tid] for tid in train_ids]) <= cleaning_capacity
            
            # Solve
            time_limit = constraints.get("solver_time_limit", 10)
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
            
            # Extract solution
            service, standby, ibl = [], [], []
            for train in trains:
                tid = train.train_id
                if pulp.value(x_ibl[tid]) and pulp.value(x_ibl[tid]) > 0.5:
                    ibl.append(tid)
                elif pulp.value(x_service[tid]) and pulp.value(x_service[tid]) > 0.5:
                    service.append(tid)
                else:
                    standby.append(tid)
            
            return {
                "service": service,
                "standby": standby,
                "ibl": ibl,
                "solver": "pulp_mip",
                "status": pulp.LpStatus[prob.status],
                "objective_value": pulp.value(prob.objective)
            }
        
        except Exception as e:
            logger.error(f"PuLP optimization failed: {e}")
            return None
    
    def _optimize_heuristic(self, trains: List[TrainData], constraints: Dict,
                           scored_df: pd.DataFrame, current_date: datetime) -> Dict:
        """Greedy heuristic optimization"""
        max_service = constraints.get("max_service_trains", 20)
        cleaning_capacity = constraints.get("cleaning_capacity_hours", 40)
        
        service, standby, ibl = [], [], []
        remaining_cleaning = cleaning_capacity
        
        for _, row in scored_df.iterrows():
            tid = row["train_id"]
            train = next((t for t in trains if t.train_id == tid), None)
            if not train:
                continue
            
            is_ready, reasons = RuleEngine.is_train_service_ready(train, current_date)
            
            if not is_ready:
                ibl.append(tid)
            elif (len(service) < max_service and 
                  remaining_cleaning >= train.cleaning_hours_required):
                service.append(tid)
                remaining_cleaning -= train.cleaning_hours_required
            else:
                standby.append(tid)
        
        return {
            "service": service,
            "standby": standby,
            "ibl": ibl,
            "solver": "greedy_heuristic",
            "status": "optimal",
            "objective_value": None
        }

# ==================== ML PREDICTION ENGINE ====================

class MLPredictionEngine:
    """Machine learning for predictive analytics"""
    
    def __init__(self):
        self.reliability_model = None
        self.health_predictor = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            if os.path.exists('reliability_model.pkl'):
                self.reliability_model = joblib.load('reliability_model.pkl')
                logger.info("Loaded reliability model")
            if os.path.exists('health_predictor.pkl'):
                self.health_predictor = joblib.load('health_predictor.pkl')
                logger.info("Loaded health predictor")
            if os.path.exists('scaler.pkl'):
                self.scaler = joblib.load('scaler.pkl')
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")
    
    def train_reliability_model(self, historical_data: pd.DataFrame):
        """Train reliability prediction model"""
        try:
            # Features: mileage, health_score, days_since_service, job_cards
            # Target: success (1) or failure (0)
            X = historical_data[['mileage_km', 'health_score', 'days_since_service', 'open_job_cards']]
            y = historical_data['success']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.reliability_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.reliability_model.fit(X_train_scaled, y_train)
            
            score = self.reliability_model.score(X_test_scaled, y_test)
            logger.info(f"Reliability model trained. Accuracy: {score:.3f}")
            
            joblib.dump(self.reliability_model, 'reliability_model.pkl')
            joblib.dump(self.scaler, 'scaler.pkl')
            
            return {"accuracy": score, "status": "success"}
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def predict_reliability(self, train: TrainData) -> float:
        """Predict service reliability for a train"""
        if not self.reliability_model:
            return 0.8  # Default
        
        try:
            days_since_service = 0
            if train.last_service_date:
                try:
                    last_service = datetime.fromisoformat(train.last_service_date)
                    days_since_service = (datetime.now() - last_service).days
                except:
                    pass
            
            features = np.array([[
                train.mileage_km,
                train.component_health_score,
                days_since_service,
                len(train.job_card_ids)
            ]])
            
            features_scaled = self.scaler.transform(features)
            prob = self.reliability_model.predict_proba(features_scaled)[0][1]
            return prob
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.8

# ==================== API ENDPOINTS ====================

# Initialize engines
scoring_engine = ScoringEngine()
optimization_engine = OptimizationEngine()
ml_engine = MLPredictionEngine()

@app.route('/health', methods=['GET'])
def health_check():
    """System health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "train_count": len(registry.get_all_trains()),
        "pulp_available": PULP_AVAILABLE
    })

@app.route('/trains', methods=['GET'])
def get_all_trains():
    """Get all registered trains"""
    trains = [t.to_dict() for t in registry.get_all_trains()]
    return jsonify({"trains": trains, "count": len(trains)})

@app.route('/trains/<train_id>', methods=['GET'])
def get_train(train_id):
    """Get specific train"""
    train = registry.get_train(train_id.upper())
    if not train:
        return jsonify({"error": "Train not found"}), 404
    return jsonify(train.to_dict())

@app.route('/trains', methods=['POST'])
def add_train():
    """Add new train (enforces unique TrainID)"""
    try:
        data = request.json
        train_id = data.get('train_id', '').strip().upper()
        
        if not train_id:
            return jsonify({"error": "TrainID is required"}), 400
        
        if registry.train_exists(train_id):
            return jsonify({"error": f"TrainID {train_id} already exists"}), 409
        
        train = TrainData(train_id)
        for key, value in data.items():
            if key != 'train_id' and hasattr(train, key):
                setattr(train, key, value)
        
        success = registry.add_train(train)
        if success:
            return jsonify({"message": "Train added successfully", "train": train.to_dict()}), 201
        else:
            return jsonify({"error": "Failed to add train"}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/trains/<train_id>', methods=['PUT'])
def update_train(train_id):
    """Update existing train"""
    try:
        data = request.json
        success = registry.update_train(train_id.upper(), data)
        if success:
            train = registry.get_train(train_id.upper())
            return jsonify({"message": "Train updated", "train": train.to_dict()})
        return jsonify({"error": "Train not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/trains/<train_id>', methods=['DELETE'])
def delete_train(train_id):
    """Delete train"""
    success = registry.remove_train(train_id.upper())
    if success:
        return jsonify({"message": "Train deleted"})
    return jsonify({"error": "Train not found"}), 404

@app.route('/trains/bulk', methods=['POST'])
def bulk_upload_trains():
    """Bulk upload trains from CSV"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        added, skipped, errors = 0, 0, []
        
        for _, row in df.iterrows():
            try:
                train_id = str(row.get('TrainID', '')).strip().upper()
                if not train_id:
                    skipped += 1
                    continue
                
                if registry.train_exists(train_id):
                    skipped += 1
                    errors.append(f"{train_id}: already exists")
                    continue
                
                train = TrainData(train_id)
                train.fitness_rolling = row.get('Fitness_Rolling', 'Valid')
                train.fitness_signal = row.get('Fitness_Signal', 'Valid')
                train.fitness_telecom = row.get('Fitness_Telecom', 'Valid')
                train.job_card_open = str(row.get('JobCard_Open', 'No')).lower() == 'yes'
                train.mileage_km = int(row.get('Mileage_km', 0))
                train.branding_contract = row.get('Branding_Contract', 'None')
                train.branding_hours_current = int(row.get('Branding_Hours_Current', 0))
                train.branding_hours_required = int(row.get('Branding_Hours_Required', 0))
                train.cleaning_hours_required = int(row.get('Cleaning_Hours_Req', 2))
                train.home_bay = row.get('Home_Bay', 'B1')
                
                if registry.add_train(train):
                    added += 1
                else:
                    skipped += 1
            
            except Exception as e:
                skipped += 1
                errors.append(f"Row error: {str(e)}")
        
        return jsonify({
            "message": "Bulk upload completed",
            "added": added,
            "skipped": skipped,
            "errors": errors[:10]  # Return first 10 errors
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/optimize', methods=['POST'])
def run_optimization():
    """Run optimization and generate induction plan"""
    try:
        params = request.json or {}
        
        # Parameters
        max_service = params.get('max_service_trains', 20)
        cleaning_capacity = params.get('cleaning_capacity_hours', 40)
        solver_time_limit = params.get('solver_time_limit', 10)
        current_date_str = params.get('current_date')
        
        if current_date_str:
            current_date = datetime.fromisoformat(current_date_str)
        else:
            current_date = datetime.now()
        
        # Get all trains
        trains = registry.get_all_trains()
        if not trains:
            return jsonify({"error": "No trains registered"}), 400
        
        # Compute context
        total_mileage = sum(t.mileage_km for t in trains)
        fleet_avg_mileage = total_mileage / len(trains)
        max_cleaning = max(t.cleaning_hours_required for t in trains)
        
        context = {
            "fleet_avg_mileage": fleet_avg_mileage,
            "max_cleaning_hours": max_cleaning
        }
        
        # Score trains
        scored_df = scoring_engine.score_all_trains(trains, context)
        
        # Run optimization
        constraints = {
            "max_service_trains": max_service,
            "cleaning_capacity_hours": cleaning_capacity,
            "solver_time_limit": solver_time_limit
        }
        
        result = optimization_engine.optimize(trains, constraints, scored_df, current_date)
        
        # Add explanations
        explanations = {}
        for train in trains:
            is_ready, reasons = RuleEngine.is_train_service_ready(train, current_date)
            state = "service" if train.train_id in result["service"] else \
                    "ibl" if train.train_id in result["ibl"] else "standby"
            
            explanations[train.train_id] = {
                "state": state,
                "service_ready": is_ready,
                "reasons": reasons if reasons else ["All checks passed"],
                "composite_score": float(scored_df[scored_df["train_id"] == train.train_id]["composite_score"].values[0]),
                "mileage_km": train.mileage_km,
                "branding_shortfall": max(0, train.branding_hours_required - train.branding_hours_current),
                "cleaning_hours": train.cleaning_hours_required,
                "health_score": train.component_health_score
            }
        
        # Calculate KPIs
        service_trains = [t for t in trains if t.train_id in result["service"]]
        total_cleaning_used = sum(t.cleaning_hours_required for t in service_trains)
        total_branding_shortfall = sum(max(0, t.branding_hours_required - t.branding_hours_current) 
                                       for t in trains)
        mileage_std = np.std([t.mileage_km for t in trains])
        avg_health = np.mean([t.component_health_score for t in service_trains]) if service_trains else 0
        
        kpis = {
            "service_count": len(result["service"]),
            "standby_count": len(result["standby"]),
            "ibl_count": len(result["ibl"]),
            "fleet_utilization": len(result["service"]) / len(trains) * 100,
            "cleaning_utilization": total_cleaning_used / cleaning_capacity * 100,
            "total_branding_shortfall_hours": total_branding_shortfall,
            "mileage_imbalance_std": float(mileage_std),
            "avg_service_health_score": float(avg_health)
        }
        
        return jsonify({
            "timestamp": current_date.isoformat(),
            "plan": result,
            "explanations": explanations,
            "kpis": kpis,
            "scored_trains": scored_df.to_dict('records')
        })
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/explain/<train_id>', methods=['GET'])
def explain_train(train_id):
    """Get detailed explanation for a specific train"""
    try:
        train = registry.get_train(train_id.upper())
        if not train:
            return jsonify({"error": "Train not found"}), 404
        
        current_date = datetime.now()
        is_ready, reasons = RuleEngine.is_train_service_ready(train, current_date)
        
        # Get all fitness checks
        fitness_valid, fitness_reasons = RuleEngine.check_fitness_validity(train, current_date)
        jobcard_valid, jobcard_reasons = RuleEngine.check_job_card_status(train)
        health_valid, health_reasons = RuleEngine.check_component_health(train)
        
        # Compute individual scores
        trains = registry.get_all_trains()
        fleet_avg_mileage = sum(t.mileage_km for t in trains) / len(trains) if trains else 0
        context = {"fleet_avg_mileage": fleet_avg_mileage, "max_cleaning_hours": 8}
        
        mileage_score = scoring_engine.compute_mileage_score(train, fleet_avg_mileage)
        branding_score = scoring_engine.compute_branding_score(train)
        cleaning_score = scoring_engine.compute_cleaning_score(train, 8)
        health_score = scoring_engine.compute_health_score(train)
        reliability_score = scoring_engine.compute_reliability_score(train)
        composite_score = scoring_engine.compute_composite_score(train, context)
        
        # ML prediction
        predicted_reliability = ml_engine.predict_reliability(train)
        
        explanation = {
            "train_id": train.train_id,
            "service_ready": is_ready,
            "overall_status": "READY" if is_ready else "NOT READY",
            "disqualification_reasons": reasons,
            "detailed_checks": {
                "fitness": {
                    "valid": fitness_valid,
                    "reasons": fitness_reasons,
                    "rolling": train.fitness_rolling,
                    "signal": train.fitness_signal,
                    "telecom": train.fitness_telecom
                },
                "job_cards": {
                    "valid": jobcard_valid,
                    "reasons": jobcard_reasons,
                    "open": train.job_card_open,
                    "critical": train.job_card_critical,
                    "ids": train.job_card_ids
                },
                "health": {
                    "valid": health_valid,
                    "reasons": health_reasons,
                    "score": train.component_health_score,
                    "iot_alerts": train.iot_sensor_alerts
                }
            },
            "scoring_breakdown": {
                "composite_score": float(composite_score),
                "mileage_score": float(mileage_score),
                "branding_score": float(branding_score),
                "cleaning_score": float(cleaning_score),
                "health_score": float(health_score),
                "reliability_score": float(reliability_score),
                "weights": scoring_engine.weights
            },
            "operational_data": {
                "mileage_km": train.mileage_km,
                "fleet_avg_mileage": fleet_avg_mileage,
                "branding_contract": train.branding_contract,
                "branding_hours_current": train.branding_hours_current,
                "branding_hours_required": train.branding_hours_required,
                "branding_shortfall": max(0, train.branding_hours_required - train.branding_hours_current),
                "cleaning_hours_required": train.cleaning_hours_required,
                "home_bay": train.home_bay,
                "current_bay": train.current_bay
            },
            "ml_predictions": {
                "predicted_reliability": float(predicted_reliability)
            }
        }
        
        return jsonify(explanation)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/whatif', methods=['POST'])
def what_if_analysis():
    """What-if scenario analysis"""
    try:
        params = request.json
        train_id = params.get('train_id', '').upper()
        changes = params.get('changes', {})
        
        if not train_id:
            return jsonify({"error": "train_id is required"}), 400
        
        train = registry.get_train(train_id)
        if not train:
            return jsonify({"error": "Train not found"}), 404
        
        # Create scenario copy
        scenario_train = TrainData(train_id)
        for key, value in train.to_dict().items():
            if hasattr(scenario_train, key):
                setattr(scenario_train, key, value)
        
        # Apply changes
        for key, value in changes.items():
            if hasattr(scenario_train, key):
                setattr(scenario_train, key, value)
        
        # Compare before/after
        current_date = datetime.now()
        
        original_ready, original_reasons = RuleEngine.is_train_service_ready(train, current_date)
        scenario_ready, scenario_reasons = RuleEngine.is_train_service_ready(scenario_train, current_date)
        
        trains = registry.get_all_trains()
        fleet_avg_mileage = sum(t.mileage_km for t in trains) / len(trains)
        context = {"fleet_avg_mileage": fleet_avg_mileage, "max_cleaning_hours": 8}
        
        original_score = scoring_engine.compute_composite_score(train, context)
        scenario_score = scoring_engine.compute_composite_score(scenario_train, context)
        
        return jsonify({
            "train_id": train_id,
            "changes_applied": changes,
            "comparison": {
                "original": {
                    "service_ready": original_ready,
                    "reasons": original_reasons,
                    "composite_score": float(original_score)
                },
                "scenario": {
                    "service_ready": scenario_ready,
                    "reasons": scenario_reasons,
                    "composite_score": float(scenario_score)
                },
                "impact": {
                    "readiness_changed": original_ready != scenario_ready,
                    "score_delta": float(scenario_score - original_score),
                    "improvement": scenario_score > original_score
                }
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/kpis', methods=['GET'])
def get_kpis():
    """Get fleet-wide KPIs and analytics"""
    try:
        trains = registry.get_all_trains()
        if not trains:
            return jsonify({"error": "No trains registered"}), 400
        
        current_date = datetime.now()
        
        # Service readiness
        ready_trains = []
        not_ready_trains = []
        for train in trains:
            is_ready, _ = RuleEngine.is_train_service_ready(train, current_date)
            if is_ready:
                ready_trains.append(train)
            else:
                not_ready_trains.append(train)
        
        # Mileage statistics
        mileages = [t.mileage_km for t in trains]
        mileage_stats = {
            "avg": float(np.mean(mileages)),
            "std": float(np.std(mileages)),
            "min": int(np.min(mileages)),
            "max": int(np.max(mileages)),
            "total": int(np.sum(mileages))
        }
        
        # Branding statistics
        branding_shortfalls = [max(0, t.branding_hours_required - t.branding_hours_current) 
                               for t in trains]
        branding_contracts = {}
        for train in trains:
            contract = train.branding_contract
            if contract not in branding_contracts:
                branding_contracts[contract] = {
                    "count": 0,
                    "total_shortfall": 0,
                    "total_current": 0,
                    "total_required": 0
                }
            branding_contracts[contract]["count"] += 1
            branding_contracts[contract]["total_shortfall"] += max(0, train.branding_hours_required - train.branding_hours_current)
            branding_contracts[contract]["total_current"] += train.branding_hours_current
            branding_contracts[contract]["total_required"] += train.branding_hours_required
        
        # Health statistics
        health_scores = [t.component_health_score for t in trains]
        health_stats = {
            "avg": float(np.mean(health_scores)),
            "min": float(np.min(health_scores)),
            "max": float(np.max(health_scores)),
            "below_threshold_count": sum(1 for h in health_scores if h < 70)
        }
        
        # Job card statistics
        open_jobcards = sum(1 for t in trains if t.job_card_open)
        critical_jobcards = sum(1 for t in trains if t.job_card_critical)
        
        # Fitness certificate status
        fitness_issues = {
            "rolling_expired": sum(1 for t in trains if t.fitness_rolling.lower() == "expired"),
            "signal_expired": sum(1 for t in trains if t.fitness_signal.lower() == "expired"),
            "telecom_expired": sum(1 for t in trains if t.fitness_telecom.lower() == "expired")
        }
        
        kpis = {
            "timestamp": datetime.utcnow().isoformat(),
            "fleet_size": len(trains),
            "service_readiness": {
                "ready_count": len(ready_trains),
                "not_ready_count": len(not_ready_trains),
                "readiness_rate": len(ready_trains) / len(trains) * 100
            },
            "mileage": mileage_stats,
            "branding": {
                "total_shortfall": sum(branding_shortfalls),
                "contracts": branding_contracts
            },
            "health": health_stats,
            "maintenance": {
                "open_jobcards": open_jobcards,
                "critical_jobcards": critical_jobcards
            },
            "fitness_certificates": fitness_issues
        }
        
        return jsonify(kpis)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("KMRL AI-Driven Train Induction Planning System")
    logger.info("Smart India Hackathon 2025 - Backend Server")
    logger.info("=" * 60)
    logger.info(f"Fleet size: {len(registry.get_all_trains())} trains")
    logger.info(f"PuLP optimization: {'Available' if PULP_AVAILABLE else 'Not available (using heuristic)'}")
    logger.info("=" * 60)
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
