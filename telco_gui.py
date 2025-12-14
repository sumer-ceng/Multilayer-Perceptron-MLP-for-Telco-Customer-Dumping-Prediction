"""
Telco MLP Projesi - GUI Görselleştirme
Deney sonuçlarını ve ağ topolojilerini görsel arayüzde gösterir.
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path


class TelcoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Telco Müşteri Terk Tahmini - MLP Sonuçları")
        self.root.geometry("1300x850")
        self.root.minsize(1100, 750)
        
        self.colors = {
            'bg': '#0f0f23', 'card': '#1e1e3f', 'accent': '#e94560',
            'text': '#ffffff', 'muted': '#8892b0', 'success': '#27ae60',
            'warning': '#f39c12', 'error': '#e74c3c', 'header': '#0f3460'
        }
        self.root.configure(bg=self.colors['bg'])
        
        self.experiments = [
            ("🧪 Deney 1: Eğitim = Test", "1_Egitim_Verisi_Test", "#e74c3c"),
            ("📊 Deney 2: 5-Fold CV", "2_5fold_CrossValidation", "#9b59b6"),
            ("📈 Deney 3: 10-Fold CV", "3_10fold_CrossValidation", "#3498db"),
            ("✂️ Deney 4: Holdout 75-25", "4_Holdout_75-25_EnBasarili", "#27ae60"),
        ]
        
        self.exp_data = {
            0: {"name": "Eğitim verisini test olarak kullanma", "desc": "Model, eğitildiği veri üzerinde test edilir.\n⚠️ Overfitting riski yüksek!",
                "params": {"Gizli Katmanlar": "(64, 32)", "Alpha (L2 Reg.)": "0.0001", "Öğrenme Oranı": "0.01"}, "acc": 0.9246, "conf": [[4901, 262], [268, 1601]]},
            1: {"name": "5-fold cross validation", "desc": "Veri 5 eşit parçaya bölünür.\nHer parça sırayla test seti olarak kullanılır.",
                "params": {"Gizli Katmanlar": "(16,)", "Alpha (L2 Reg.)": "0.001", "Öğrenme Oranı": "0.001"}, "acc": 0.7969, "conf": [[4621, 542], [886, 983]]},
            2: {"name": "10-fold cross validation", "desc": "Veri 10 eşit parçaya bölünür.\n✓ En güvenilir değerlendirme yöntemi!",
                "params": {"Gizli Katmanlar": "(16,)", "Alpha (L2 Reg.)": "0.001", "Öğrenme Oranı": "0.001"}, "acc": 0.7991, "conf": [[4626, 537], [876, 993]]},
            3: {"name": "Holdout %75-%25 (En başarılı)", "desc": "Veri rastgele %75 eğitim, %25 test olarak ayrılır.\n5 farklı rastgele ayırma yapılır.",
                "params": {"Gizli Katmanlar": "(16,)", "Alpha (L2 Reg.)": "0.0001", "Öğrenme Oranı": "0.001"}, "acc": 0.8100, "conf": [[1165, 126], [208, 259]]},
        }
        
        self.setup_ui()
        self.show_experiment(0)
    
    def setup_ui(self):
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        header = tk.Frame(main, bg=self.colors['card'], height=90)
        header.pack(fill=tk.X, pady=(0, 15))
        header.pack_propagate(False)
        
        tk.Label(header, text="🧠 Telco Müşteri Terk Tahmini", font=('Segoe UI', 22, 'bold'),
                bg=self.colors['card'], fg=self.colors['text']).pack(side=tk.LEFT, padx=25, pady=20)
        
        info = tk.Frame(header, bg=self.colors['card'])
        info.pack(side=tk.RIGHT, padx=25, pady=15)
        tk.Label(info, text="📚 Yapay Sinir Ağlarına Giriş  |  👤 Ahmet Sümer  |  🔢 221213028",
                font=('Segoe UI', 10), bg=self.colors['card'], fg=self.colors['muted']).pack()
        
        # Content
        content = tk.Frame(main, bg=self.colors['bg'])
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left Panel
        left = tk.Frame(content, bg=self.colors['card'], width=420)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)
        
        tk.Label(left, text="🔬 Deney Seçimi", font=('Segoe UI', 12, 'bold'),
                bg=self.colors['header'], fg=self.colors['text']).pack(fill=tk.X, pady=12, ipady=5)
        
        btn_frame = tk.Frame(left, bg=self.colors['card'])
        btn_frame.pack(fill=tk.X, padx=15, pady=15)
        
        for i, (label, _, color) in enumerate(self.experiments):
            btn = tk.Button(btn_frame, text=label, font=('Segoe UI', 11, 'bold'),
                           bg=color, fg='white', activebackground=color, bd=0,
                           cursor='hand2', command=lambda x=i: self.show_experiment(x))
            btn.pack(fill=tk.X, pady=5, ipady=10)
        
        tk.Label(left, text="📋 Deney Sonuçları", font=('Segoe UI', 12, 'bold'),
                bg=self.colors['accent'], fg=self.colors['text']).pack(fill=tk.X, pady=(15, 0), ipady=8)
        
        self.result_frame = tk.Frame(left, bg=self.colors['bg'])
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right Panel
        right = tk.Frame(content, bg=self.colors['card'])
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        tk.Label(right, text="🎨 Ağ Topolojisi Görselleştirmesi", font=('Segoe UI', 12, 'bold'),
                bg=self.colors['header'], fg=self.colors['text']).pack(fill=tk.X, pady=12, ipady=5)
        
        self.img_frame = tk.Frame(right, bg=self.colors['bg'])
        self.img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.img_label = tk.Label(self.img_frame, bg=self.colors['bg'])
        self.img_label.pack(expand=True)
        
        # Footer
        footer = tk.Frame(main, bg=self.colors['card'], height=35)
        footer.pack(fill=tk.X, pady=(15, 0))
        tk.Label(footer, text="💡 Bir deney seçerek sonuçları ve ağ yapısını görüntüleyin  |  🔧 telco.py ile modelleri yeniden eğitin",
                font=('Segoe UI', 9), bg=self.colors['card'], fg=self.colors['muted']).pack(pady=8)
    
    def show_experiment(self, idx):
        exp = self.exp_data[idx]
        col = self.experiments[idx][2]
        
        # Sonuçları temizle
        for w in self.result_frame.winfo_children():
            w.destroy()
        
        # Deney adı
        tk.Label(self.result_frame, text=exp['name'], font=('Segoe UI', 11, 'bold'),
                bg=col, fg='white', pady=8, padx=10).pack(fill=tk.X, pady=(0, 10))
        
        # Açıklama
        tk.Label(self.result_frame, text=exp['desc'], font=('Segoe UI', 10),
                bg=self.colors['card'], fg=self.colors['muted'], pady=8, justify='left').pack(fill=tk.X)
        
        # Doğruluk
        acc_col = self.colors['success'] if exp['acc'] > 0.8 else self.colors['warning'] if exp['acc'] > 0.7 else self.colors['error']
        acc_frame = tk.Frame(self.result_frame, bg=self.colors['card'])
        acc_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(acc_frame, text="✓ BAŞARI ORANI", font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['muted']).pack()
        tk.Label(acc_frame, text=f"{exp['acc']*100:.2f}%", font=('Segoe UI', 32, 'bold'),
                bg=self.colors['card'], fg=acc_col).pack()
        
        # İlerleme çubuğu
        prog = tk.Canvas(acc_frame, width=320, height=15, bg=self.colors['card'], highlightthickness=0)
        prog.pack(pady=5)
        prog.create_rectangle(0, 3, 320, 12, fill=self.colors['bg'], outline="")
        prog.create_rectangle(0, 3, int(320*exp['acc']), 12, fill=acc_col, outline="")
        
        # Parametreler
        tk.Label(self.result_frame, text="📋 EN İYİ PARAMETRELER", font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['muted']).pack(pady=(10, 5))
        
        for k, v in exp['params'].items():
            row = tk.Frame(self.result_frame, bg=self.colors['card'])
            row.pack(fill=tk.X, padx=15)
            tk.Label(row, text=f"• {k}:", font=('Consolas', 9), bg=self.colors['card'], fg=self.colors['muted']).pack(side=tk.LEFT)
            tk.Label(row, text=v, font=('Consolas', 9, 'bold'), bg=self.colors['card'], fg='#4ECDC4').pack(side=tk.LEFT, padx=5)
        
        # Konfüzyon matrisi
        tk.Label(self.result_frame, text="📊 KONFÜZYON MATRİSİ", font=('Segoe UI', 9),
                bg=self.colors['card'], fg=self.colors['muted']).pack(pady=(15, 8))
        
        matrix = tk.Canvas(self.result_frame, width=250, height=110, bg=self.colors['card'], highlightthickness=0)
        matrix.pack()
        
        conf = exp['conf']
        labels, colors_m = [['TN', 'FP'], ['FN', 'TP']], [[self.colors['success'], self.colors['error']], [self.colors['error'], self.colors['success']]]
        sx, sy, cw, ch = 55, 10, 70, 45
        
        matrix.create_text(sx + cw, 5, text="Tahmin", fill=self.colors['muted'], font=('Segoe UI', 8))
        matrix.create_text(20, sy + ch, text="Gerçek", fill=self.colors['muted'], font=('Segoe UI', 8), angle=90)
        
        for i in range(2):
            for j in range(2):
                x, y = sx + j*cw, sy + i*ch
                matrix.create_rectangle(x, y, x+cw, y+ch, fill=colors_m[i][j], outline=self.colors['bg'])
                matrix.create_text(x+cw//2, y+ch//2-8, text=str(conf[i][j]), fill='white', font=('Segoe UI', 12, 'bold'))
                matrix.create_text(x+cw//2, y+ch//2+10, text=labels[i][j], fill='white', font=('Segoe UI', 8))
        
        # Görsel yükle
        img_path = Path(f"telco_ag_deney{idx+1}_{self.experiments[idx][1]}.png")
        if img_path.exists():
            self.load_image(img_path)
        else:
            self.img_label.config(text="⚠️ Görsel bulunamadı\n\nÖnce çalıştırın: python telco.py",
                                 font=('Segoe UI', 11), fg=self.colors['warning'], bg=self.colors['bg'])
    
    def load_image(self, path):
        try:
            img = Image.open(path)
            self.img_frame.update()
            w, h = self.img_frame.winfo_width() - 20, self.img_frame.winfo_height() - 20
            w, h = max(w, 550), max(h, 400)
            
            ratio = min(w/img.width, h/img.height)
            new_size = (int(img.width*ratio), int(img.height*ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=photo, text="")
            self.img_label.image = photo
        except Exception as e:
            self.img_label.config(text=f"❌ Hata: {e}", font=('Segoe UI', 10), fg=self.colors['error'])


if __name__ == "__main__":
    root = tk.Tk()
    app = TelcoGUI(root)
    root.mainloop()
