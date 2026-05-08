import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import time

st.set_page_config(page_title="Detección de Anomalías en Red", layout="wide")

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(random_state=42, contamination=0.1)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_network_data(self, n_samples=1000, anomaly_percentage=0.1):
        """Genera datos de tráfico de red con anomalías controladas"""
        np.random.seed(42)
        
        n_anomalies = int(n_samples * anomaly_percentage)
        n_normal = n_samples - n_anomalies
        
        # Datos normales
        normal_data = {
            'bytes_sent': np.random.normal(500, 100, n_normal),
            'bytes_received': np.random.normal(800, 150, n_normal),
            'packet_count': np.maximum(0, np.random.normal(50, 15, n_normal)),
            'duration': np.random.normal(120, 30, n_normal),
            'connections_per_hour': np.random.normal(10, 3, n_normal),
            'error_rate': np.random.beta(2, 50, n_normal)
        }
        
        # Datos anómalos
        anomaly_data = {
            'bytes_sent': np.random.normal(2000, 500, n_anomalies),
            'bytes_received': np.random.normal(5000, 1000, n_anomalies),
            'packet_count': np.maximum(0, np.random.normal(500, 100, n_anomalies)),
            'duration': np.random.normal(5, 2, n_anomalies),
            'connections_per_hour': np.random.normal(100, 20, n_anomalies),
            'error_rate': np.random.beta(10, 2, n_anomalies)
        }
        
        normal_df = pd.DataFrame(normal_data)
        anomaly_df = pd.DataFrame(anomaly_data)
        
        normal_df['anomaly'] = 0
        anomaly_df['anomaly'] = 1
        normal_df['type'] = 'Normal'
        anomaly_df['type'] = 'Anómalo'
        
        df = pd.concat([normal_df, anomaly_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def train_model(self, data):
        """Entrena el modelo de detección de anomalías"""
        features = data[['bytes_sent', 'bytes_received', 'packet_count', 
                        'duration', 'connections_per_hour', 'error_rate']]
        
        X_scaled = self.scaler.fit_transform(features)
        self.model.fit(X_scaled)
        self.is_trained = True
        
        predictions = self.model.predict(X_scaled)
        data['predicted_anomaly'] = [1 if x == -1 else 0 for x in predictions]
        data['anomaly_score'] = self.model.decision_function(X_scaled)
        
        return data

def main():
    st.title("🌐 Dashboard de Detección de Anomalías en Red")
    st.markdown("---")
    
    # Inicializar detector
    if 'detector' not in st.session_state:
        st.session_state.detector = AnomalyDetector()
    
    # Sidebar para controles
    st.sidebar.header("🎛️ Configuración de Anomalías")
    
    anomaly_percentage = st.sidebar.slider(
        "Porcentaje de Anomalías en la Red",
        min_value=0.0,
        max_value=0.5,
        value=0.1,
        step=0.05,
        help="Controla la cantidad de tráfico anómalo en la red"
    )
    
    n_samples = st.sidebar.slider(
        "Número de Muestras de Tráfico",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100
    )
    
    # Botones de control
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("🔄 Generar Datos"):
            with st.spinner("Generando datos de tráfico..."):
                st.session_state.data = st.session_state.detector.generate_network_data(
                    n_samples, anomaly_percentage
                )
                st.success("¡Datos generados!")
    
    with col2:
        if st.button("🤖 Entrenar Modelo"):
            if 'data' in st.session_state:
                with st.spinner("Entrenando modelo ML..."):
                    st.session_state.data = st.session_state.detector.train_model(
                        st.session_state.data
                    )
                st.success("¡Modelo entrenado!")
            else:
                st.warning("Primero genera los datos")
    
    with col3:
        if st.button("📊 Actualizar Dashboard"):
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Instrucciones:**
    1. Ajusta el porcentaje de anomalías
    2. Haz clic en 'Generar Datos'
    3. Entrena el modelo ML
    4. Observa las detecciones en tiempo real
    """)
    
    # Main dashboard
    if 'data' in st.session_state:
        data = st.session_state.data
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_traffic = len(data)
            st.metric("Tráfico Total", f"{total_traffic:,}")
        
        with col2:
            real_anomalies = data['anomaly'].sum()
            st.metric("Anomalías Reales", f"{real_anomalies}")
        
        with col3:
            if 'predicted_anomaly' in data.columns:
                detected_anomalies = data['predicted_anomaly'].sum()
                st.metric("Anomalías Detectadas", f"{detected_anomalies}")
            else:
                st.metric("Anomalías Detectadas", "0")
        
        with col4:
            if 'predicted_anomaly' in data.columns:
                accuracy = (data['anomaly'] == data['predicted_anomaly']).mean() * 100
                st.metric("Precisión ML", f"{accuracy:.1f}%")
            else:
                st.metric("Precisión ML", "0%")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribución de Tráfico")
            
            # Gráfico de bytes enviados vs recibidos
            fig = px.scatter(
                data,
                x='bytes_sent',
                y='bytes_received',
                color='type' if 'predicted_anomaly' not in data.columns else 'predicted_anomaly',
                title="Bytes Enviados vs Recibidos",
                color_continuous_scale='reds' if 'predicted_anomaly' in data.columns else None,
                labels={'predicted_anomaly': 'Anomalía Detectada'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Comparación: Real vs Detectado")
            
            if 'predicted_anomaly' in data.columns:
                comparison_data = pd.DataFrame({
                    'Tipo': ['Normales', 'Anómalos'],
                    'Reales': [
                        len(data[data['anomaly'] == 0]),
                        len(data[data['anomaly'] == 1])
                    ],
                    'Detectados': [
                        len(data[data['predicted_anomaly'] == 0]),
                        len(data[data['predicted_anomaly'] == 1])
                    ]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Reales', x=comparison_data['Tipo'], y=comparison_data['Reales']))
                fig.add_trace(go.Bar(name='Detectados', x=comparison_data['Tipo'], y=comparison_data['Detectados']))
                fig.update_layout(barmode='group', title="Comparación Real vs Detectado")
                st.plotly_chart(fig, use_container_width=True)
        
        # Gráficos adicionales
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Conexiones por Hora")
            fig = px.histogram(
                data,
                x='connections_per_hour',
                color='type' if 'predicted_anomaly' not in data.columns else 'predicted_anomaly',
                nbins=30,
                title="Distribución de Conexiones por Hora"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚡ Tasa de Error vs Duración")
            fig = px.scatter(
                data,
                x='duration',
                y='error_rate',
                color='type' if 'predicted_anomaly' not in data.columns else 'predicted_anomaly',
                title="Duración vs Tasa de Error",
                size='packet_count'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de datos
        st.subheader("📋 Datos de Tráfico")
        display_cols = ['bytes_sent', 'bytes_received', 'connections_per_hour', 'error_rate', 'type']
        if 'predicted_anomaly' in data.columns:
            display_cols.extend(['predicted_anomaly', 'anomaly_score'])
        
        st.dataframe(data[display_cols].head(20), use_container_width=True)
        
    else:
        # Pantalla de bienvenida
        st.info("👈 Usa la barra lateral para generar datos y comenzar el análisis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 ¿Cómo funciona?")
            st.markdown("""
            1. **Ajusta el porcentaje de anomalías** - Controla cuánto tráfico malicioso hay
            2. **Genera datos** - Crea tráfico de red simulado
            3. **Entrena el modelo ML** - Isolation Forest detecta patrones anómalos
            4. **Analiza resultados** - Ve detecciones en tiempo real
            """)
        
        with col2:
            st.subheader("🔧 Tecnologías Utilizadas")
            st.markdown("""
            - **Machine Learning**: Isolation Forest
            - **Visualización**: Plotly (gráficos interactivos)
            - **Framework**: Streamlit
            - **Procesamiento**: Scikit-learn, Pandas
            """)

if __name__ == "__main__":
    main()