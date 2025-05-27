#!/usr/bin/env python3
import time
import os

def test_buzzer_sysfs():
    """Test del buzzer usando sysfs directamente"""
    
    # Pin 32 = GPIO12
    gpio_num = 12
    
    try:
        print("🔧 Configurando GPIO...")
        
        # Export GPIO (ignorar si ya existe)
        try:
            with open('/sys/class/gpio/export', 'w') as f:
                f.write(str(gpio_num))
            print(f"✅ GPIO {gpio_num} exportado")
        except:
            print(f"⚠️  GPIO {gpio_num} ya exportado")
        
        # Set direction to output
        with open(f'/sys/class/gpio/gpio{gpio_num}/direction', 'w') as f:
            f.write('out')
        print(f"✅ GPIO {gpio_num} configurado como salida")
        
        # Test: hacer parpadear el pin
        print("🔊 Enviando señal de prueba (3 segundos)...")
        
        for i in range(6):  # 3 segundos de parpadeo
            # HIGH
            with open(f'/sys/class/gpio/gpio{gpio_num}/value', 'w') as f:
                f.write('1')
            time.sleep(0.25)
            
            # LOW  
            with open(f'/sys/class/gpio/gpio{gpio_num}/value', 'w') as f:
                f.write('0')
            time.sleep(0.25)
            
            print(f"  Pulso {i+1}/6")
        
        print("✅ Test básico completado")
        return True
        
    except PermissionError:
        print("❌ Error de permisos. Ejecuta con sudo:")
        print("sudo python3 quick_buzzer_test.py")
        return False
    except FileNotFoundError as e:
        print(f"❌ GPIO sysfs no encontrado: {e}")
        print("¿Estás en Jetson Nano con GPIO habilitado?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_buzzer_tone():
    """Test con tono real"""
    gpio_num = 12
    frequency = 440  # A4
    duration = 2     # segundos
    
    try:
        print(f"🎵 Generando tono {frequency}Hz por {duration}s...")
        
        # Export y configurar (ignorar errores)
        try:
            with open('/sys/class/gpio/export', 'w') as f:
                f.write(str(gpio_num))
        except:
            pass
            
        with open(f'/sys/class/gpio/gpio{gpio_num}/direction', 'w') as f:
            f.write('out')
        
        # Generar tono
        period = 1.0 / frequency
        half_period = period / 2.0
        end_time = time.time() + duration
        
        pulse_count = 0
        while time.time() < end_time:
            with open(f'/sys/class/gpio/gpio{gpio_num}/value', 'w') as f:
                f.write('1')
            time.sleep(half_period)
            
            with open(f'/sys/class/gpio/gpio{gpio_num}/value', 'w') as f:
                f.write('0')
            time.sleep(half_period)
            pulse_count += 1
        
        print(f"✅ Tono completado ({pulse_count} pulsos)")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        # Cleanup seguro
        try:
            with open(f'/sys/class/gpio/gpio{gpio_num}/value', 'w') as f:
                f.write('0')
        except:
            pass

def cleanup_gpio():
    """Limpiar GPIO al final"""
    gpio_num = 12
    try:
        with open(f'/sys/class/gpio/gpio{gpio_num}/value', 'w') as f:
            f.write('0')
        with open('/sys/class/gpio/unexport', 'w') as f:
            f.write(str(gpio_num))
        print("🧹 GPIO limpiado")
    except:
        print("⚠️  Error limpiando GPIO (normal si no estaba exportado)")

if __name__ == "__main__":
    print("🎵 TEST DEL BUZZER - JETSON NANO (SYSFS)")
    print("=" * 45)
    
    print("\n📋 CONEXIÓN DEL BUZZER:")
    print("  Positivo (+) → Pin 32 (GPIO12)")
    print("  Negativo (-) → Pin 30 o 34 (GND)")
    print("  Tipo: Buzzer activo (no pasivo)")
    
    success = False
    
    print("\n1. Test básico de GPIO:")
    if test_buzzer_sysfs():
        success = True
        
        print("\n2. Test de tono:")
        test_buzzer_tone()
    
    print("\n3. Limpieza:")
    cleanup_gpio()
    
    if success:
        print("\n🎉 ¡BUZZER FUNCIONA! Tu line follower debería sonar.")
        print("Ejecuta: python3 tu_line_follower.py")
    else:
        print("\n❌ Buzzer no funciona. Revisa:")
        print("  - Conexiones físicas")
        print("  - Tipo de buzzer (activo vs pasivo)")
        print("  - Ejecutar con sudo si hay problemas de permisos")