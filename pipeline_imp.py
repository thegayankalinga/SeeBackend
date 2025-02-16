from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd


class ProjectEffortPipeline:
    def __init__(self):
        # Define feature names in order
        self.feature_names = [
            'region', 'dev_environment', 'sit_environment', 'uat_environment',
            'staging_environment', 'training_environment', 'production_environment',
            'dr_environment', 'compliance_pci_sff', 'compliance_country_specific',
            'backend_technology', 'frontend_technology', 'mobile_technology',
            'database', 'google_sso', 'apple_sso', 'facebook_sso', 'iam_vendor',
            'infrastructure_type', 'dependency_complexity', 'customer_decision_speed',
            'client_technical_knowledge', 'device_test_coverage', 'test_automation',
            'regression_type', 'middleware_availability', 'payment_provider_integration',
            'fido', 'data_migration',
            'no_of_languages', 'no_of_rtl_languages', 'tps_required',
            'warranty_months', 'no_of_functional_modules',
            'no_of_none_functional_modules', 'uat_cycles', 'test_coverage',
            'rest_integration_points', 'soap_integration_points',
            'iso8583_integration_points', 'sdk_integration_points'
        ]

        # Define column types
        self.num_cols = [
            'no_of_languages', 'no_of_rtl_languages', 'tps_required',
            'warranty_months', 'no_of_functional_modules',
            'no_of_none_functional_modules', 'uat_cycles', 'test_coverage',
            'rest_integration_points', 'soap_integration_points',
            'iso8583_integration_points', 'sdk_integration_points'
        ]

        self.cat_cols = [
            'region', 'dev_environment', 'sit_environment', 'uat_environment',
            'staging_environment', 'training_environment', 'production_environment',
            'dr_environment', 'compliance_pci_sff', 'compliance_country_specific',
            'backend_technology', 'frontend_technology', 'mobile_technology',
            'database', 'google_sso', 'apple_sso', 'facebook_sso', 'iam_vendor',
            'infrastructure_type', 'dependency_complexity', 'customer_decision_speed',
            'client_technical_knowledge', 'device_test_coverage', 'test_automation',
            'regression_type', 'middleware_availability', 'payment_provider_integration',
            'fido', 'data_migration'
        ]

        self.target_cols = ['delivery', 'engineering', 'devops', 'qa']

        # Initialize transformers
        self.num_pipeline = None
        self.cat_pipeline = None
        self.target_scaler = None
        self.label_encoders = {}

    def fit(self, df):
        """Fit the pipeline on training data"""
        # Create a copy of the DataFrame to avoid modifying the original
        df = df.copy()

        # Numerical pipeline
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        for col in self.cat_cols:
            le = LabelEncoder()
            self.label_encoders[col] = le
            # Convert to string and ensure consistent formatting for categorical values
            df[col] = df[col].apply(lambda x: str(int(x)) if isinstance(x, (int, float)) else str(x))
            df[col] = le.fit_transform(df[col])

        # Fit numerical pipeline
        if self.num_cols:
            self.num_pipeline.fit(df[self.num_cols])

        # Fit target scaler
        self.target_scaler = StandardScaler()
        if all(col in df.columns for col in self.target_cols):
            self.target_scaler.fit(df[self.target_cols])

        # Drop project_name column at the end if it exists
        if 'project_name' in df.columns:
            df = df.drop(columns=['project_name'])

        return self

    def transform(self, features):
        """Transform input features"""
        # Convert features list to DataFrame with correct column names
        if isinstance(features, dict) and 'features' in features:
            features = features['features']

        if len(features) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(features)}")

        df = pd.DataFrame([features], columns=self.feature_names)

        # Transform categorical features
        for col in self.cat_cols:
            if col in df.columns:
                # Convert to string and ensure consistent formatting
                df[col] = df[col].apply(lambda x: str(int(x)) if isinstance(x, (int, float)) else str(x))
                df[col] = self.label_encoders[col].transform(df[col])

        # Transform numerical features
        if self.num_cols:
            df[self.num_cols] = self.num_pipeline.transform(df[self.num_cols])

        return df

    def inverse_transform_targets(self, predictions):
        """Transform predictions back to original scale"""
        return self.target_scaler.inverse_transform(predictions)