import serial
import time

# ── Konfiguration ──────────────────────────────────────────────────────────────
PORT        = "/dev/ttyAMA0"
BAUDRATE    = 115200
LINE_END    = "\n"          # anpassen: "\r\n" falls nötig
ENCODING    = "ascii"       # anpassen: "utf-8" falls nötig
TIMEOUT_S   = 2.0           # Sekunden auf Antwort warten
DELAY_S     = 1.0           # Sekunden zwischen zwei Befehlen

# ── Befehle (Raspi → Tiny) ─────────────────────────────────────────────────────
CMD_PUZZLE      = "puzzlesteps"
CMD_PLACING     = "placingsteps"
CMD_ADJUSTMENT  = "adjustmentsteps"
CMD_DROP        = "drop"
CMD_STOP        = "STOP"

# ── Erwartete Antworten (Tiny → Raspi) ────────────────────────────────────────
ACK_PUZZLE      = "puzzlesteps"
ACK_PLACING     = "placingsteps"
ACK_ADJUSTMENT  = "adjustmentsteps"
ACK_DROP        = "drop"
ACK_NOT_IMPL    = "command not implemented"
ACK_STOP_RECV   = "STOP erkannt"
ACK_STOP_DONE   = "Befehl STOP ausgefuehrt"

# ── Testsequenz: (Befehl, X, Y, K) ────────────────────────────────────────────
TEST_SEQUENCE = [
    (CMD_PUZZLE,     100,  200,  45),
    (CMD_PLACING,    300,  150,   0),
    (CMD_ADJUSTMENT,  10,  -10,   5),
    (CMD_DROP,         0,    0,   0),
    (CMD_PUZZLE,    -100, -200, -45),   # signed Test
    (CMD_STOP,         0,    0,   0),
]

# ── Hilfsfunktionen ────────────────────────────────────────────────────────────
def send(ser: serial.Serial, cmd: str, x: int, y: int, k: int) -> str:
    msg = f"{cmd} {x} {y} {k}"
    raw = (msg + LINE_END).encode(ENCODING)
    ser.write(raw)
    print(f"  → Sent:     {msg!r}")

    response = ser.readline().decode(ENCODING).strip()
    print(f"  ← Received: {response!r}")
    return response

# ── Hauptprogramm ──────────────────────────────────────────────────────────────
def main():
    print(f"Öffne {PORT} @ {BAUDRATE} Baud")
    with serial.Serial(PORT, baudrate=BAUDRATE, timeout=TIMEOUT_S) as ser:
        time.sleep(0.5)  # warten bis UART bereit
        ser.reset_input_buffer()

        for i, (cmd, x, y, k) in enumerate(TEST_SEQUENCE):
            print(f"\n[{i+1}/{len(TEST_SEQUENCE)}] {cmd}")
            send(ser, cmd, x, y, k)

            if cmd == CMD_STOP:
                # STOP schickt zwei Antworten
                second = ser.readline().decode(ENCODING).strip()
                print(f"  ← Received: {second!r}")

            if i < len(TEST_SEQUENCE) - 1:
                print(f"  ... warte {DELAY_S}s")
                time.sleep(DELAY_S)

    print("\nTest abgeschlossen.")

if __name__ == "__main__":
    main()