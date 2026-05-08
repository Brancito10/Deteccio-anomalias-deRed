import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SIEM con ML", layout="wide")

class SIEMSystem:
    def __init__(self):
        self.events_log = []
        self.attack_types = {
            'ddos': 'Ataque DDoS',
            'brute_force': 'Fuerza Bruta',
            'malware': 'Infección Malware',
            'phishing': 'Ataque Phishing',
            'insider': 'Amenaza Interna',
            'scanning': 'Escaneo de Puertos'
        }
        self.ml_model = IsolationForest(random_state=42, contamination=0.15)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def generate_normal_event(self):
        """Genera un evento normal de red"""
        event_types = ['login', 'file_access', 'network_connection', 'process_start', 'dns_query']
        users = [f'user{i:02d}' for i in range(1, 11)]
        departments = ['IT', 'HR', 'Finance', 'Sales', 'Engineering']
        
        return {
            'timestamp': datetime.now(),
            'event_type': np.random.choice(event_types),
            'user': np.random.choice(users),
            'source_ip': f"192.168.1.{np.random.randint(1, 255)}",
            'department': np.random.choice(departments),
            'severity': 'Low',
            'description': 'Actividad normal del sistema',
            'is_attack': False,
            'attack_type': 'None'
        }
    
    def generate_attack_event(self, attack_type):
        """Genera un evento de ataque específico"""
        users = [f'user{i:02d}' for i in range(1, 11)]
        
        base_event = {
            'timestamp': datetime.now(),
            'user': np.random.choice(users),
            'source_ip': f"10.0.0.{np.random.randint(1, 255)}",
            'is_attack': True,
            'attack_type': attack_type
        }
        
        if attack_type == 'ddos':
            base_event.update({
                'event_type': 'network_flood',
                'severity': 'Critical',
                'description': f'Posible ataque DDoS desde IP externa'
            })
        elif attack_type == 'brute_force':
            base_event.update({
                'event_type': 'failed_login',
                'severity': 'High',
                'description': 'Múltiples intentos de login fallidos'
            })
        elif attack_type == 'malware':
            base_event.update({
                'event_type': 'suspicious_process',
                'severity': 'High',
                'description': 'Ejecución de proceso sospechoso detectado'
            })
        elif attack_type == 'phishing':
            base_event.update({
                'event_type': 'email_alert',
                'severity': 'Medium',
                'description': 'Correo phishing detectado en bandeja de entrada'
            })
        elif attack_type == 'insider':
            base_event.update({
                'event_type': 'unauthorized_access',
                'severity': 'High',
                'description': 'Acceso no autorizado a archivos confidenciales'
            })
        elif attack_type == 'scanning':
            base_event.update({
                'event_type': 'port_scan',
                'severity': 'Medium',
                'description': 'Escaneo de puertos detectado en la red'
            })
        
        return base_event
    
    def add_event(self, is_attack=False, attack_type=None):
        """Añade un evento al log"""
        if is_attack and attack_type:
            event = self.generate_attack_event(attack_type)
        else:
            event = self.generate_normal_event()
        
        self.events_log.append(event)
        return event
    
    def train_ml_model(self):
        """Entrena el modelo ML para detección de anomalías"""
        if len(self.events_log) < 10:
            return None
        
        df = pd.DataFrame(self.events_log)
        
        # Convertir características a numéricas
        df['severity_num'] = df['severity'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4})
        df['hour'] = df['timestamp'].dt.hour
        df['is_external'] = df['source_ip'].apply(lambda x: 1 if x.startswith('10.0.0.') else 0)
        
        # Codificar tipos de evento
        event_types = df['event_type'].unique()
        event_mapping = {event: i for i, event in enumerate(event_types)}
        df['event_type_num'] = df['event_type'].map(event_mapping)
        
        features = df[['severity_num', 'hour', 'is_external', 'event_type_num']].fillna(0)
        
        if len(features) > 0:
            X_scaled = self.scaler.fit_transform(features)
            self.ml_model.fit(X_scaled)
            self.is_trained = True
            
            # Añadir predicciones al dataframe
            predictions = self.ml_model.predict(X_scaled)
            df['ml_anomaly'] = [1 if x == -1 else 0 for x in predictions]
            df['ml_confidence'] = np.abs(self.ml_model.decision_function(X_scaled))
            
            return df
        
        return None

def main():
    st.title("🛡️ SIEM con Machine Learning")
    st.markdown("---")
    
    # Inicializar SIEM
    if 'siem' not in st.session_state:
        st.session_state.siem = SIEMSystem()
        st.session_state.auto_mode = False
    
    siem = st.session_state.siem
    
    # Layout principal
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🎮 Panel de Control")
        
        # Selector de tipo de ataque
        attack_type = st.selectbox(
            "Tipo de Ataque a Simular",
            list(siem.attack_types.keys()),
            format_func=lambda x: siem.attack_types[x]
        )
        
        # Botones de control
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🚀 Agregar Ataque", use_container_width=True):
                event = siem.add_event(is_attack=True, attack_type=attack_type)
                st.success(f"Ataque {siem.attack_types[attack_type]} agregado!")
        
        with col_btn2:
            if st.button("✅ Agregar Evento Normal", use_container_width=True):
                siem.add_event(is_attack=False)
                st.success("Evento normal agregado!")
        
        st.markdown("---")
        
        # Modo automático
        auto_mode = st.checkbox("Modo Automático", value=st.session_state.auto_mode)
        if auto_mode != st.session_state.auto_mode:
            st.session_state.auto_mode = auto_mode
            st.rerun()
        
        if st.session_state.auto_mode:
            auto_speed = st.slider("Velocidad (eventos/seg)", 1, 10, 3)
            if st.button("⏹️ Detener Auto", use_container_width=True):
                st.session_state.auto_mode = False
                st.rerun()
        
        st.markdown("---")
        
        # Botón de entrenamiento ML
        if st.button("🤖 Entrenar Modelo ML", use_container_width=True):
            if len(siem.events_log) >= 10:
                with st.spinner("Entrenando modelo..."):
                    st.session_state.ml_results = siem.train_ml_model()
                st.success("Modelo ML entrenado!")
            else:
                st.warning("Necesitas al menos 10 eventos para entrenar")
        
        # Estadísticas rápidas
        st.subheader("📊 Stats Rápidas")
        total_events = len(siem.events_log)
        attack_events = len([e for e in siem.events_log if e['is_attack']])
        
        st.metric("Eventos Totales", total_events)
        st.metric("Ataques Simulados", attack_events)
        if total_events > 0:
            st.metric("Tasa de Ataques", f"{(attack_events/total_events)*100:.1f}%")
    
    with col1:
        # Dashboard principal
        st.subheader("📈 Dashboard en Tiempo Real")
        
        if siem.events_log:
            df = pd.DataFrame(siem.events_log)
            
            # Métricas principales
            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
            
            with col_met1:
                total_events = len(df)
                st.metric("Total Eventos", total_events)
            
            with col_met2:
                attacks = df['is_attack'].sum()
                st.metric("Ataques Detectados", attacks)
            
            with col_met3:
                if 'ml_anomaly' in df.columns:
                    ml_detections = df['ml_anomaly'].sum()
                    st.metric("Alertas ML", ml_detections)
                else:
                    st.metric("Alertas ML", "0")
            
            with col_met4:
                critical_events = len(df[df['severity'] == 'Critical'])
                st.metric("Críticos", critical_events)
            
            # Gráficos
            tab1, tab2, tab3 = st.tabs(["📊 Distribución", "🕒 Timeline", "🔍 Detecciones ML"])
            
            with tab1:
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Gráfico de tipos de evento
                    event_counts = df['event_type'].value_counts()
                    fig = px.pie(
                        values=event_counts.values,
                        names=event_counts.index,
                        title="Distribución de Tipos de Evento"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_chart2:
                    # Gráfico de severidad
                    severity_counts = df['severity'].value_counts()
                    fig = px.bar(
                        x=severity_counts.index,
                        y=severity_counts.values,
                        title="Eventos por Nivel de Severidad",
                        color=severity_counts.index,
                        color_discrete_map={
                            'Low': 'green', 
                            'Medium': 'yellow', 
                            'High': 'orange', 
                            'Critical': 'red'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Timeline de eventos
                df_timeline = df.copy()
                df_timeline['time'] = df_timeline['timestamp'].dt.strftime('%H:%M:%S')
                df_timeline['color'] = df_timeline['is_attack'].apply(lambda x: 'red' if x else 'blue')
                
                fig = px.scatter(
                    df_timeline,
                    x='timestamp',
                    y='severity',
                    color='is_attack',
                    size=[10] * len(df_timeline),
                    title="Timeline de Eventos",
                    color_discrete_map={True: 'red', False: 'blue'},
                    hover_data=['event_type', 'description']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                if 'ml_anomaly' in df.columns:
                    col_ml1, col_ml2 = st.columns(2)
                    
                    with col_ml1:
                        # Detecciones ML vs Reales
                        ml_comparison = pd.DataFrame({
                            'Tipo': ['Normales', 'Anómalos'],
                            'Reales': [
                                len(df[df['is_attack'] == False]),
                                len(df[df['is_attack'] == True])
                            ],
                            'ML Detectados': [
                                len(df[df['ml_anomaly'] == 0]),
                                len(df[df['ml_anomaly'] == 1])
                            ]
                        })
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name='Reales', x=ml_comparison['Tipo'], y=ml_comparison['Reales']))
                        fig.add_trace(go.Bar(name='ML Detectados', x=ml_comparison['Tipo'], y=ml_comparison['ML Detectados']))
                        fig.update_layout(title="ML vs Detecciones Reales")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_ml2:
                        # Matriz de confusión simple
                        true_positives = len(df[(df['is_attack'] == True) & (df['ml_anomaly'] == 1)])
                        false_positives = len(df[(df['is_attack'] == False) & (df['ml_anomaly'] == 1)])
                        
                        confusion_data = [[true_positives, false_positives], [0, 0]]  # Simplificado
                        fig = px.imshow(
                            confusion_data,
                            labels=dict(x="Predicción", y="Real", color="Count"),
                            x=['Anómalo', 'Normal'],
                            y=['Ataque', 'Normal'],
                            title="Matriz de Confusión (Simplificada)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Entrena el modelo ML para ver las detecciones")
            
            # Tabla de eventos recientes
            st.subheader("📋 Eventos Recientes")
            display_df = df[['timestamp', 'event_type', 'severity', 'description', 'is_attack']].tail(10)
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.info("👈 Usa el panel de control para agregar eventos y ataques")
    
    # Modo automático
    if st.session_state.auto_mode and 'auto_speed' in locals():
        time.sleep(1.0 / auto_speed)
        # Agregar evento aleatorio (90% normal, 10% ataque)
        if np.random.random() < 0.1:
            attack_type = np.random.choice(list(siem.attack_types.keys()))
            siem.add_event(is_attack=True, attack_type=attack_type)
        else:
            siem.add_event(is_attack=False)
        st.rerun()

if __name__ == "__main__":
    main()