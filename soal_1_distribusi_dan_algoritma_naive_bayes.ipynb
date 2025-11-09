{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uYFqQaVbeqAW"
      },
      "source": [
        "# PERHATIAN!\n",
        "\n",
        "Soal yang Anda kerjakan akan dinilai secara otomatis oleh sistem dari Dicoding. Oleh karena itu, Anda **tidak diperbolehkan mengubah nama fungsi yang sudah ditentukan**, karena hal tersebut dapat memengaruhi proses penilaian.\n",
        "\n",
        "Pastikan Anda membaca dan memperhatikan setiap instruksi dengan saksama serta menuliskan kode di antara tanda komentar yang telah disediakan.\n",
        "\n",
        "> **# MULAI KODE DI SINI**\n",
        "\n",
        "\n",
        "> **# AKHIRI KODE DI SINI**\n",
        "\n",
        "---\n",
        "Selain itu, Anda **tidak diperkenankan menggunakan *library* di luar yang telah ditentukan**. Seluruh soal telah dirancang agar dapat diselesaikan dengan *library* yang tersedia.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FR7BmDQTBxXQ"
      },
      "source": [
        "# Algoritma Naive Bayes\n",
        "Data kini memegang peran yang sangat penting dalam berbagai bidang, termasuk data science dan machine learning. Data dapat dimanfaatkan tidak hanya untuk menghasilkan wawasan dan mendukung pengambilan keputusan berbasis fakta, tetapi juga untuk mengembangkan produk-produk inovatif yang berdampak nyata.\n",
        "\n",
        "\n",
        "Dalam permasalahan kali ini, Anda akan bekerja dengan data yang menerapkan prinsip-prinsip probabilitas dan statistika dengan menyelesaikan permasalahan sederhana, yakni memprediksi jenis/ras burung menggunakan algoritma Naive Bayes.\n",
        "\n",
        "Anda akan membuat data sintetis untuk jenis-jenis burung tersebut dengan mengikuti prinsip-prinsip probabilitas sehingga pada permasalahan kali ini akan dibagi dalam dua segmen.\n",
        "\n",
        "## Segmen:\n",
        "1. **Menghasilkan Data Acak**: Belajar menghasilkan data acak yang mengikuti distribusi tertentu.\n",
        "2. **Klasifikasi Naive Bayes**: Mengimplementasikan klasifikasi Naive Bayes dari data yang dihasilkan pada segmen 1.\n",
        "\n",
        "Jangan khawatir! Anda akan diberikan panduan untuk mengerjakan submission ini.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "b0wT_4CVT8t8"
      },
      "outputs": [],
      "source": [
        "# TIDAK DIPERBOLEHKAN MENGGUNAKAN LIBRARY LAIN\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from scipy.stats import uniform, binom, norm\n",
        "from scipy.special import erfinv\n",
        "from dataclasses import dataclass"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "75IQ3CzXH21G"
      },
      "source": [
        "# Segmen 1: Menghasilkan Data Acak\n",
        "Mari rekap beberapa konsep serta materi yang telah Anda pelajari selama belajar di kelas Matematika untuk Data Science.\n",
        "\n",
        "Dalam dunia data science, Anda akan sering berhadapan dengan konsep populasi dan sampel. Populasi dan sampel tidak hanya merujuk pada entitas atau kumpulan individu, tetapi juga dapat menggambarkan kumpulan kejadian atau fenomena tertentu.\n",
        "\n",
        "Saat menjumpai kasus yang berhubungan dengan suatu event/kejadian, Anda tidak hanya ingin tahu \"berapa peluang sebuah kejadian terjadi\", tetapi juga ingin melihat keseluruhan pola distribusi dari nilai-nilai yang mungkin.\n",
        "\n",
        "Dalam melihat pola tersebut, Anda bisa menggunakan fungsi distribusi yang berperan untuk memahami probabilitas tersebar dalam ruang kemungkinan.\n",
        "\n",
        "Anda sudah mempelajari tiga jenis fungsi distribusi yang umum dipakai, yakni\n",
        "- **Probability Mass Function (PMF)**: Digunakan untuk mencari distribusi terhadap variabel diskrit.\n",
        "- **Probability Density Function (PDF)**: Digunakan untuk mencari distribusi terhadap nilai kontinu.\n",
        "- **Cummulative Distribution Function (CDF)**: Digunakan untuk melihat total peluang yang sudah terkumpul sampai suatu nilai tertentu, bisa digunakan untuk variabel diskret dan kontinu\n",
        "\n",
        "\n",
        "Dalam segmen pertama, Anda akan diajak untuk menghasilkan data acak berdasarkan ketiga jenis distribusi tersebut.\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "idf5V-XbVZXx"
      },
      "source": [
        "## Variabel Acak\n",
        "\n",
        "Sebuah variabel acak (dinotasikan sebagai $X$) adalah fungsi yang mewakili fenomena acak, artinya nilai yang dihasilkan tidak bisa dipastikan tepat, tetapi memberikan peluang untuk berbagai nilai yang bisa diambil.\n",
        "\n",
        "Contohnya, jika $X$ adalah variabel acak kontinu dengan distribusi uniform [2,4], kita tidak bisa menentukan nilai pasti $X$, tetapi kita bisa mengatakan bahwa nilai $X$ berada pada rentang 2 sampai 4.\n",
        "\n",
        "> Perlu diingat: Variabel acak kontinu, mencari peluang tepat pada angka tertentu (misalnya angka 3) hampir tidak mungkin dan bisa dikatakan nol. Hal ini disebabkan, saat Anda mencari angka 3 pada rentang 2 sampai 4. Anda bisa mendapatkan angka 2.1, 2.001, atau bahkan 2.000000001. Jumlah nilai yang mungkin tidak terhitung (tak terhingga). Oleh karena itu, peluang variabel acak kontinu pada satu titik tunggal = 0. Peluang yang bermakna adalah peluang dalam suatu invterval.\n",
        "\n",
        "Konsep di atas adalah yang disebut sebagai **probability density function** (PDF) yang secara matematis memiliki rumus berikut.\n",
        "$$\n",
        "P(a \\leq X \\leq b) = \\int_a^b f(x) \\, dx\n",
        "$$\n",
        "\n",
        "Artinya, peluang $X$ berada dalam rentang [a,b] dihitung sebagai luas di bawah kurva $f(x)$ antara $a$ dan $b$.\n",
        "\n",
        "Pada variabel acak diskret, kita mengenal **probability mass function** (PMF), yaitu fungsi yang menunjukkan probabilitas untuk setiap nilai yang mungkin diambil oleh variabel diskret.\n",
        "\n",
        "$$\n",
        "P(X = x) = p_x\n",
        "$$\n",
        "\n",
        "Berbeda dengan variabel acak kontinu yang memiliki jumlah nilai tak terhingga dalam rentang tertentu (sehingga peluang tepat di satu nilai adalah nol), variabel acak diskret hanya memiliki sejumlah nilai yang terhitung sehingga bisa memberikan peluang positif pada setiap nilai tersebut.\n",
        "\n",
        "Contohnya adalah kasus pelemparan dadu. Saat melempar sebuah dadu bersisi enam, peluang munculnya setiap angka (1 hingga 6) sama, yaitu 1/6.\n",
        "\n",
        "$$\n",
        "P(X = k) = \\frac{1}{6}, \\quad k = 1, 2, 3, 4, 5, 6.\n",
        "$$\n",
        "\n",
        "Fungsi lain yang terkait dengan variabel acak adalah **cummulative distribution function** (CDF) yang dinotasikan dengan $F$. Ini mewakili peluang bahwa variabel acak $X$ kurang dari atau sama dengan x, untuk setiap x dalam bilangan riil.\n",
        "\n",
        "Artinya, jika Anda memiliki angka x, $F(x)$ memberi tahu Anda **seberapa besar peluang semua nilai di bawah atau sama dengan x.**\n",
        "\n",
        "Contohnya dalam kasus pelemparan dadu, dalam PMF kita mengetahui bahwa peluang mendapatkan angka 1 sampai 6 adalah sama.\n",
        "- $P(X = 1) = 1/6$\n",
        "- $P(X = 2) = 1/6$\n",
        "- $P(X = 3) = 1/6$\n",
        "- $ ... $\n",
        "\n",
        "CDF-nya memberikan informasi semua peluang sampai titik yang diinginkan (misalnya 6).\n",
        "- $P(X ≤ 1) = 1/6$\n",
        "- $P(X ≤ 2) = 1/6 + 1/6 = 2/6$\n",
        "- $P(X ≤ 3) = 3/6$\n",
        "- $ ... $\n",
        "- $P(X ≤ 6) = 6/6$\n",
        "\n",
        "> Materi PDF, CMF, dan PDF dapat diakses dalam materi distribusi probabilitas pada modul ketiga.\n",
        "\n",
        "Ada hal menarik yang harus Anda pahami, kita bisa menghasilkan data buatan dengan distribusi yang kita inginkan, jika kita tahu cummulative distribution function-nya ($F$).\n",
        "\n",
        "Hal ini disebabkan karena jika $X$ adalah variabel acak dengan CDF F, $F(X)$ mengikuti distribusi uniform di antara 0 dan 1. Dengan kata lain, variabel acak baru $F(X)$ akan terdistribusi uniform antara 0 dan 1. Ini membuka kemungkinan menghasilkan data buatan dengan distribusi yang kita inginkan, jika kita tahu F.\n",
        "\n",
        "Proses untuk mencapai hal tersebut adalah berikut.\n",
        "1. Hasilkan nilai acak $y$ yang terdistribusi uniform dari interval [0,1]\n",
        "2. Hitung $F^{-1}(y)$, yaitu fungsi inverse dari F yang dievaluasi pada y.\n",
        "\n",
        "Dapat dibuktikan bahwa jika $Y$ mengikuti distribusi uniform antara 0 dan 1, variabel acak $F^{-1}(y)$ memiliki distribusi yang sama dengan $X$. Dengan begitu, mengitung invers dari F bisa menghasilkan data buatan dari distribusi mana pun yang diketahui.\n",
        "\n",
        "Pada materi di kelas, Anda sudah mengenal banyak jenis distribusi probabilitas. Kali ini, kita akan coba menghasilkan angka acak mengikuti tiga jenis distribusi probabilitas berikut.\n",
        "\n",
        "*   Uniform Distribution\n",
        "*   Binomial Distribution\n",
        "*   Gaussian Distribution\n",
        "\n",
        "Mari kita mulai dengan menulis kode untuk membuat nilai acak untuk masing-masing distribusi di atas.\n",
        "\n",
        "> Pro Tips: Mengapa kita perlu mengetahui variabel acak dalam statistik? Dalam dunia data science, setiap dataset yang Anda olah adalah sebuah variabel acak. Penting untuk memahami jenis dataset yang Anda miliki untuk bisa mengetahui distribusi dan karakteristiknya.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KzSi8n_QdpE9"
      },
      "source": [
        "### Tugas 1: Membuat Data Distribusi Uniform.\n",
        "\n",
        "Distribusi uniform adalah distribusi peluang ketika semua nilai dalam rentang tertentu sama-sama mungkin terjadi. Dengan asumsi $a ≤ x ≤ b$, distribusi uniform memiliki rumus sebagai berikut.\n",
        "\n",
        "$$\n",
        "P(X=x) = 1/(b-a)\n",
        "$$\n",
        "\n",
        "Dengan variabel yang diketahui\n",
        "- $x$: angka yang ingin dihitung.\n",
        "- $b$: batas atas rentang yang diketahui.\n",
        "- $a$: batas bawah rentang yang diketahui."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "L2NlVKf4s1pB"
      },
      "source": [
        "#### Tugas 1.1: Membuat Fungsi untuk Menghasilkan Angka Acak Uniform\n",
        "\n",
        "\n",
        "Sekarang, mari buat sebuah fungsi yang bertujuan untuk membuat angka acak yang mengikuti distribusi uniform.\n",
        "\n",
        "> Tips: Anda dapat melihat referensi fungsi dari [numpy.random](https://numpy.org/doc/2.2/reference/random/index.html) untuk menghasilkan data random berbentuk uniform.\n",
        "\n",
        "**INGAT! TIDAK DIPERKENANKAN** untuk mengubah nama fungsi yang sudah didefinisikan."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "qmV9PtQZH11c"
      },
      "outputs": [],
      "source": [
        "def generate_rand_uniform(lower_bound:float, upper_bound:float, num_samples:int):\n",
        "  \"\"\"\n",
        "  Fungsi ini ditujukan untuk menghasilkan array bilangan acak yang terdistribusi uniform dalam rentang yang ditentukan.\n",
        "\n",
        "  Parameters:\n",
        "  - upper_bound (float): Batas bawah rentang\n",
        "  - lower_bound (float): Batas atas rentang\n",
        "  - num_samples (int): Jumlah sampel yang dihasilkan\n",
        "\n",
        "  Return:\n",
        "  - array (ndarray): Array bilangan acak yang terdistribusi uniform pada rentang [a,b)\n",
        "\n",
        "  \"\"\"\n",
        "  np.random.seed(15) #JANGAN UBAH KODE INI\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "  # Anda wajib menggunakan numpy.random.uniform (dari numpy) untuk menjaga konsistensi output.\n",
        "  array = np.random.uniform(lower_bound, upper_bound, num_samples)\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return array"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "vu6tN61n-AQA"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "7 nilai acak yang diambil dari interval 0 sampai 1:\n",
            " [0.849 0.179 0.054 0.362 0.275 0.53  0.306]\n",
            "\n",
            "12 nilai acak yang diambil dari interval 20 sampai 70:\n",
            " [62.441 28.945 22.718 38.077 33.77  46.5   35.296 35.224 25.587 32.495\n",
            " 65.881 33.207]\n",
            "\n",
            "2 nilai acak yang diambil dari interval 10 sampai 99:\n",
            " [85.545 25.922]\n",
            "\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "print(f\"7 nilai acak yang diambil dari interval 0 sampai 1:\\n {np.array2string(generate_rand_uniform(0,1, num_samples=7), precision=3)}\\n\")\n",
        "print(f\"12 nilai acak yang diambil dari interval 20 sampai 70:\\n {np.array2string(generate_rand_uniform(20,70, num_samples=12), precision=3)}\\n\")\n",
        "print(f\"2 nilai acak yang diambil dari interval 10 sampai 99:\\n {np.array2string(generate_rand_uniform(10,99, num_samples=2), precision=3)}\\n\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3K54et4f_Vdl"
      },
      "source": [
        "##### Output yang diharapkan\n",
        "\n",
        "```\n",
        "7 nilai acak yang diambil dari interval 0 sampai 1:\n",
        " [0.849 0.179 0.054 0.362 0.275 0.53  0.306]\n",
        "\n",
        "12 nilai acak yang diambil dari interval 20 sampai 70:\n",
        " [62.441 28.945 22.718 38.077 33.77  46.5   35.296 35.224 25.587 32.495\n",
        " 65.881 33.207]\n",
        "\n",
        "2 nilai acak yang diambil dari interval 10 sampai 99:\n",
        " [85.545 25.922]\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YCFFEYApOlzR"
      },
      "source": [
        "#### Tugas 1.2: Membuat Visualisasi Distribusi Uniform\n",
        "\n",
        "Hebat! Fungsi untuk menghasilkan data acak dengan bentuk distribusi uniform telah berhasil dibuat!\n",
        "\n",
        "Sekarang, kita akan melihat visualisasi dari data yang dihasilkan berdasarkan distribusi uniform tersebut. Anda hanya perlu fokus pada dua tugas yang sudah didefinisikan dalam fungsi.\n",
        "\n",
        "\n",
        "**Catatan:**\n",
        "\n",
        "Fungsi ini tidak akan masuk dalam pengujian, silakan untuk berkreasi hingga memenuhi ekspektasi yang diharapkan."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "OUAOoyMCJ2ur"
      },
      "outputs": [],
      "source": [
        "def viz_uniform_teoretis(lower_bound, upper_bound, num_samples):\n",
        "  \"\"\"\n",
        "  Menghasilkan visualisasi distribusi uniform.\n",
        "\n",
        "  Parameters:\n",
        "  - upper_bound (float): Batas bawah rentang\n",
        "  - lower_bound (float): Batas atas rentang\n",
        "  - num_samples (int): Jumlah sampel yang dihasilkan\n",
        "  \"\"\"\n",
        "\n",
        "  # Mengatur ukuran plot\n",
        "  plt.figure(figsize=(8,5))\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "\n",
        "  # Tugas 1: Gunakan fungsi generate_rand_uniform dengan parameter yang sama\n",
        "  # dengan fungsi visualisasi ini, simpan hasilnya ke dalam sebuah variabel.\n",
        "  data = generate_rand_uniform(lower_bound, upper_bound, num_samples)\n",
        "\n",
        "\n",
        "  # Tugas 2: Buatlah sebuah histogram dari variabel sebelumnya dengan parameter-parameternya sebagai berikut.\n",
        "  # - Total bins adalah 30.\n",
        "  # - Mengaktifkan probability density\n",
        "  # - Menggunakan warna lightgrey\n",
        "  # - Menggunakan edgecolor hitam\n",
        "  plt.hist(data, bins=30, density=True, color='lightgrey', edgecolor='black')\n",
        "\n",
        "\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  x = np.linspace(lower_bound - 2, upper_bound + 2, 500)\n",
        "  pdf = uniform.pdf(x, loc=lower_bound, scale=upper_bound - lower_bound)\n",
        "  plt.plot(x, pdf, color='blue')\n",
        "  plt.fill_between(x, pdf, 0, where=(x >= lower_bound) & (x <= upper_bound), color='skyblue', alpha=0.3)\n",
        "\n",
        "  plt.axvline(lower_bound, color='grey', linestyle='--', label=f\"Batas Bawah (lower_bound={lower_bound})\")\n",
        "  plt.axvline(upper_bound, color='grey', linestyle='--', label=f\"Batas Atas (upper_bound={upper_bound})\")\n",
        "\n",
        "  plt.title(\"Distribusi Uniform\")\n",
        "  plt.legend()\n",
        "  plt.savefig(\"uniform-viz.png\")\n",
        "  plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "pushOJSNqgYV"
      },
      "outputs": [
        {
          "data": {
            "image/png": "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",
            "text/plain": [
              "<Figure size 800x500 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "# JANGAN mengubah kode di bawah ini.\n",
        "viz_uniform_teoretis(lower_bound=0, upper_bound=10, num_samples=10000000)\n",
        "\n",
        "\n",
        "# Pro Tips:\n",
        "# Dalam statistik, semakin banyak jumlah sampel, hasilnya akan semakin baik dan mendekati karakteristik populasi sebenarnya.\n",
        "# Anda dapat mencobanya dengan mengubah jumlah sampel menjadi lebih sedikit\n",
        "# daripada contoh di atas. Semakin kecil jumlah sampel, distribusi pada histogram\n",
        "# akan terlihat lebih acak dan tidak merata. Sebaliknya, semakin besar jumlah sampel,\n",
        "# bentuk histogram akan semakin mendekati distribusi uniform yang diharapkan.\n",
        "\n",
        "# Gunakan ini untuk melihat visualisasi lainnya\n",
        "# viz_uniform_teoretis(lower_bound=0, upper_bound=10, num_samples=100000)\n",
        "# viz_uniform_teoretis(lower_bound=0, upper_bound=10, num_samples=100)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "x0nOIPDkzKxG"
      },
      "source": [
        "##### Output yang diharapkan\n",
        "\n",
        "<img src=\"https://drive.google.com/uc?id=1cVxtTj2c9iOEszz3pxCDHaC-s7IbG0Jj\" style=\"height:300px;\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AjlvbHrgEkPR"
      },
      "source": [
        "### Tugas 2: Membuat Data Distribusi Gaussian\n",
        "\n",
        "Sekarang Anda telah memiliki sebuah fungsi yang dapat menghasilkan data acak dengan distribusi uniform. Sebagaimana yang dijelaskan sebelumnya, kita dapat membentuk variabel dengan distribusi tertentu jika memiliki angka-angka yang berdistribusi uniform.\n",
        "\n",
        "Mari kita buat variabel-variabel acak yang mengikuti distribusi Gaussian dengan masuk ke tahap berikutnya, yaitu menghitung inverse CDF $F^{-1}(y)$."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hHH74OnrF-OB"
      },
      "source": [
        "#### Tugas 2.1: Distribusi Gaussian - Inverse CDF\n",
        "\n",
        "Distribusi Gaussian juga dikenal sebagai distribusi normal. Distribusi ini menggambarkan pola penyebaran data dari begitu banyak fenomena. Sayangnya, tidak ada bentuk rumus tertutup (closed-form) untuk fungsi CDF dari distribusi normal.\n",
        "\n",
        "Dalam distribusi Gaussian, perhitungannya melibatkan fungsi yang disebut Gaussian Error Function, yang dinotasikan sebagai $erf(x)$. Kita dapat membuat fungsi inverse CDF untuk distribusi normal dengan memanfaatkan error function tersebut.\n",
        "\n",
        "Namun, implementasi proses ini cukup rumit jika dilakukan secara manual (dan di luar jangkauan pembelajaran kelas). Untuk itu, Anda dapat memanfaatkan pustaka Python, seperti `scipy.special.erf`, `scipy.special.erfinv`, atau `math.erf` untuk mengimplementasikannya dengan lebih mudah.\n",
        "\n",
        "Jika sebuah variabel acak mengikuti distribusi normal $X \\sim N(\\mu, \\sigma)$, CDF dapat diekspresikan sebagai berikut.\n",
        "\n",
        "$$y = F(x) = \\frac{1}{2} \\left[ 1 + \\text{erf}\\left( \\frac{x - \\mu}{\\sigma \\sqrt{2}} \\right) \\right]$$\n",
        "\n",
        "Namun, jika kita mengambil invers dari fungsi erf, rumusnya dapat dituliskan sebagai berikut.\n",
        "\n",
        "$$x = F^{-1}(y) = \\sigma \\sqrt{2} \\cdot \\text{erf}^{-1}(2y - 1) + \\mu\n",
        "$$\n",
        "\n",
        "\n",
        "> Catatan:\n",
        "- Miu (μ): Nilai rata-rata dari distribusi normal/Gaussian.\n",
        "- Sigma (σ): Nilai Standar Deviasi dari distribusi normal/Gaussian.\n",
        "- $F$: Cummulative Distribution Function (CDF).\n",
        "\n",
        "> Tips:\n",
        "- Gunakan rumus di atas untuk fungsi uniform_inverse_cdf\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "BSpGnmBp-Y11"
      },
      "outputs": [],
      "source": [
        "def uniform_inverse_cdf(probability, miu, sigma):\n",
        "  \"\"\"\n",
        "  Menghitung inverse cummulative distribution function (CDF) dari distribusi Gaussian.\n",
        "\n",
        "  Parameters:\n",
        "  - probability (float atau ndarray): Probabilitas atau array dari probabilitas.\n",
        "  - miu (float): Rata-rata dari distribusi Gaussian.\n",
        "  - sigma (float): Standar deviasi dari distribusi Gaussian.\n",
        "\n",
        "  Return:\n",
        "  - x (float atau ndarray): Nilai inverse CDF dari distribusi Gaussian berdasarkan probabilitas yang diberikan.\n",
        "  \"\"\"\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "  x = miu + sigma * np.sqrt(2) * erfinv(2 * probability - 1)\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return x"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "sodEhgg3UkYk"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Inverse CDF dari distribusi Gaussian dengan miu 15 dan sigma 5 dengan angka 1e-07: -10.997\n",
            "Inverse CDF dari distribusi Gaussian dengan miu 15 dan sigma 5 dengan angka 1: inf\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "print(f\"Inverse CDF dari distribusi Gaussian dengan miu {15} dan sigma {5} dengan angka {1e-7}: {uniform_inverse_cdf(1e-7, 15, 5):.3f}\")\n",
        "print(f\"Inverse CDF dari distribusi Gaussian dengan miu {15} dan sigma {5} dengan angka {1}: {uniform_inverse_cdf(1, 15, 5):.3f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "S_pIyrQeVvGO"
      },
      "source": [
        "##### Output yang diharapkan\n",
        "\n",
        "```\n",
        "Inverse CDF dari distribusi Gaussian dengan miu 15 dan sigma 5 dengan angka 1e-07: -10.997\n",
        "Inverse CDF dari distribusi Gaussian dengan miu 15 dan sigma 5 dengan angka 1: inf\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Pf8L8PrxWY_3"
      },
      "source": [
        "#### Tugas 2.2: Menghasilkan Gaussian Distribution\n",
        "\n",
        "Keren! Anda sudah memiliki fungsi yang bisa menghitung inverse CDF untuk distribusi Gaussian. Selanjutnya, mari kita gabungkan dua fungsi yang sudah dibuat tersebut untuk membuat data acak dengan distribusi Gaussian."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "XVPtTroTVmSU"
      },
      "outputs": [],
      "source": [
        "def generate_gaussian(miu, sigma, num_samples):\n",
        "  \"\"\"\n",
        "  Menghasilkan data dengan distribusi Gaussian/Normal.\n",
        "\n",
        "  Parameters:\n",
        "  - miu (float): Rata-rata dari distribusi Gaussian.\n",
        "  - sigma (float): Standar deviasi dari distribusi Gaussian.\n",
        "  - num_samples (int): Jumlah sampel yang dihasilkan.\n",
        "\n",
        "  Returns:\n",
        "  - array (ndarray): Array dari probabilitas distribusi Gaussian.\n",
        "  \"\"\"\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "\n",
        "  # Petunjuk:\n",
        "  # - Gunakan fungsi generate_rand_uniform untuk menghasilkan data acak uniform dengan rentang 0 dan 1.\n",
        "  array = miu + sigma * np.sqrt(2) * erfinv(2 * generate_rand_uniform(0, 1, num_samples) - 1)\n",
        "\n",
        "  # Petunjuk:\n",
        "  # - Gunakan fungsi uniform_inverse_cdf untuk melakukan inverse CDF dari data acak yang dihasilkan pada fungsi sebelumnya.\n",
        "\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return array"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "u6GInLy7Y3xm"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "3 angka pertama untuk variabel first_gaussian: [ 1.03137584 -0.91958083 -1.60394386]\n",
            "3 angka pertama untuk variabel second_gaussian: [18.18825505  6.48251504  2.37633686]\n",
            "3 angka pertama untuk variabel third_gaussian: [15.15687921  5.40209587  1.98028072]\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "first_gaussian = generate_gaussian(0,1,1000)\n",
        "second_gaussian = generate_gaussian(12,6,1000)\n",
        "third_gaussian = generate_gaussian(10,5,1000)\n",
        "\n",
        "print(f\"3 angka pertama untuk variabel first_gaussian: {first_gaussian[:3]}\")\n",
        "print(f\"3 angka pertama untuk variabel second_gaussian: {second_gaussian[:3]}\")\n",
        "print(f\"3 angka pertama untuk variabel third_gaussian: {third_gaussian[:3]}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5yIumSJvdqLS"
      },
      "source": [
        "##### Output yang diharapkan\n",
        "\n",
        "```\n",
        "3 angka pertama untuk variabel first_gaussian: [ 1.03137584 -0.91958083 -1.60394386]\n",
        "3 angka pertama untuk variabel second_gaussian: [18.18825505  6.48251504  2.37633686]\n",
        "3 angka pertama untuk variabel third_gaussian: [15.15687921  5.40209587  1.98028072]\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8zQkXi1Cd0zs"
      },
      "source": [
        "#### Tugas 2.3: Membuat Visualisasi Distribusi Gaussian\n",
        "\n",
        "Mantap! Angka distribusi Gaussian telah berhasil didapatkan!\n",
        "\n",
        "Selanjutnya, Anda perlu membuat kembali visualisasi yang ditujukan untuk menghasilkan distribusi Gaussian.\n",
        "\n",
        "**Catatan:**\n",
        "\n",
        "Fungsi ini tidak akan masuk dalam pengujian, silakan untuk berkreasi hingga memenuhi ekspektasi yang diharapkan."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "z06j86rqZNON"
      },
      "outputs": [],
      "source": [
        "def viz_distribusi_gaussian(first_gaussian, second_gaussian, third_gaussian):\n",
        "  \"\"\"\n",
        "  Menghasilkan histogram plot untuk tiga distribusi Gaussian yang dihasilkan pada pengujian.\n",
        "\n",
        "  Parameters:\n",
        "  - first_gaussian (ndarray): array dari distribusi Gaussian pertama.\n",
        "  - second_gaussian (ndarray): array dari distribusi Gaussian kedua.\n",
        "  - third_gaussian (ndarray): array dari distribusi Gaussian ketiga.\n",
        "  \"\"\"\n",
        "  fig, ax = plt.subplots(1, 1, figsize=(10, 4))\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "\n",
        "  # Tugas:\n",
        "  # Buatlah tiga buah histogram plot yang diambil dari variabel first_gaussian, second_gaussian, dan third_gaussian dengan menggunakan parameter-parameter berikut.\n",
        "  # - alpha=0.5,\n",
        "  # - Jumlah bins adalah 32.\n",
        "  # - Menggunakan label sesuai jenis distribusi Gaussian-nya.\n",
        "  ax.hist(first_gaussian, bins=32, alpha=0.5, label='Gaussian (miu=0, sigma=1)')\n",
        "  ax.hist(second_gaussian, bins=32, alpha=0.5, label='Gaussian (miu=12, sigma=6)')\n",
        "  ax.hist(third_gaussian, bins=32, alpha=0.5, label='Gaussian (miu=10, sigma=5)')\n",
        "\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  ax.set_title(\"Histograms of Gaussian distributions\")\n",
        "  ax.set_xlabel(\"Values\")\n",
        "  ax.set_ylabel(\"Frequencies\")\n",
        "  ax.legend()\n",
        "  plt.savefig(\"gaussian-viz.png\")\n",
        "  plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "id": "QOHfSvs7ZIaK"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA0oAAAGHCAYAAACOM6KuAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABkmElEQVR4nO3deVyVZf7/8fdREAFB3ABJRFxQVFxJRUvRRMsl13S0wq3U0HLJDcvEUkgdl8rQcSp1ptGsUZum1MRcchITt0zNJUVciXLDDRG5f3/443w7h0VA8CC+no/HeYznuq/7uj/31pwP13Vft8kwDEMAAAAAALMStg4AAAAAAIoaEiUAAAAAsEKiBAAAAABWSJQAAAAAwAqJEgAAAABYIVECAAAAACskSgAAAABghUQJAAAAAKyQKAEAAACAFRIlAEXG0qVLZTKZtGvXriyXd+nSRdWqVbMoq1atmgYOHJin7Wzfvl0RERG6fPly/gJFlt58801VrVpVdnZ2cnNzu2f9//3vf+rXr5+qVq0qBwcHOTs7q169enr99dd1+PDhwg+4AGRcsydPnrR1KGbBwcEKDg42fz958qRMJpOWLl2ap3aWL1+u+fPn52mdrLYVEREhk8mkP/74I09t5eTQoUOKiIjI8rgPHDgw038nACA/SJQAPNTWrFmjKVOm5Gmd7du3a9q0aSRKBeg///mPZsyYodDQUG3dulUbN27Msf6bb76pJ598UgkJCXrzzTe1fv16ffnllxo8eLBiYmLk7++vO3fuPKDo869z586KjY1V5cqVbR1KtipXrqzY2Fh17tw5T+vlJ1HK77by6tChQ5o2bVqWidKUKVO0Zs2aQt0+gEeDna0DAID70bhxY1uHkGe3b9+WyWSSnV3x+U/wgQMHJEmvvfaa3N3dc6y7YsUKzZgxQ8OHD1d0dLRMJpN5WUhIiMaOHavo6OhCjbegVKpUSZUqVbJ1GDlycHBQixYtCnUbd+7cUVpa2gPZ1r3UqFHDptsHUHzQowTgoWY99C49PV3Tp09X7dq15ejoKDc3NzVo0EDvvfeepLvDgMaPHy9J8vX1lclkkslk0pYtW8zrz5o1S3Xq1JGDg4Pc3d0VGhqqM2fOWGzXMAxFRkbKx8dHpUuXVmBgoGJiYjINe9qyZYtMJpP++c9/6vXXX9djjz0mBwcH/frrr/r9998VFhamunXrqkyZMnJ3d1e7du20bds2i21lDGeaPXu2Zs6cqWrVqsnR0VHBwcE6evSobt++rUmTJsnLy0tly5ZVjx49lJSUZNHGpk2bFBwcrAoVKsjR0VFVq1ZVr169dOPGjRyPb26OR7Vq1fTmm29Kkjw8PGQymRQREZFtm9OnT1fFihU1b948iyQpg8lk0ogRI1SyZElzWUxMjLp166YqVaqodOnSqlmzpoYNG5ZpOFd2w64yhn/92RdffKHmzZurbNmycnJyUvXq1TV48GCLfc/pWpKyHnqX21gzYjp48KD69eunsmXLysPDQ4MHD9aVK1eyPX4ZDMPQrFmzzNdgkyZNtG7dukz1shoO9/vvv2vo0KHy9vaWg4ODKlWqpFatWpl7AoODg/XNN98oISHBfI9kHL+M9mbNmqXp06fL19dXDg4O2rx5c47D/E6fPq2ePXvK1dVVZcuW1QsvvKDff//dok52186f7/OlS5fqueeekyS1bdvWHFvGNrO6BlJSUhQeHi5fX1+VKlVKjz32mEaMGJGpV7latWrq0qWL1q9fryZNmsjR0VF16tTRJ598YlHvxo0bGjdunHx9fVW6dGmVL19egYGBWrFiRabYATy8is+fMwEUGxl/nbZmGMY91501a5YiIiL05ptvqnXr1rp9+7YOHz5s/kH00ksv6eLFi/rggw+0evVq85CpunXrSpJeeeUVLV68WCNHjlSXLl108uRJTZkyRVu2bNGePXtUsWJFSdIbb7yhqKgoDR06VD179tTp06f10ksv6fbt2/Lz88sUV3h4uIKCgrRo0SKVKFFC7u7u5h+JU6dOlaenp65du6Y1a9YoODhY3333nUXCJUkffvihGjRooA8//FCXL1/W66+/rq5du6p58+ayt7fXJ598ooSEBI0bN04vvfSSvvrqK0l3f9h27txZTz75pD755BO5ubnp7NmzWr9+vVJTU+Xk5JTt8czN8VizZo0+/PBDffzxx1q/fr3Kli2rKlWqZNneuXPndOjQIfXr10+lS5e+5/nMcPz4cQUFBemll15S2bJldfLkSc2dO1dPPPGEfv75Z9nb2+e6LUmKjY1V37591bdvX0VERKh06dJKSEjQpk2bzHXudS0VVKy9evVS3759NWTIEP38888KDw+XpEw/zq1NmzZN06ZN05AhQ9S7d2+dPn1aL7/8su7cuaPatWvnuO6LL76oPXv2aMaMGfLz89Ply5e1Z88eXbhwQZIUHR2toUOH6vjx49kOY3v//ffl5+env/71r3J1dVWtWrVy3GaPHj3Up08fDR8+XAcPHtSUKVN06NAh/fjjj3k6f507d1ZkZKQmT56sDz/8UE2aNJGUfU+SYRjq3r27vvvuO4WHh+vJJ5/U/v37NXXqVMXGxio2NlYODg7m+j/99JNef/11TZo0SR4eHvroo480ZMgQ1axZU61bt5YkjR07Vv/85z81ffp0NW7cWNevX9eBAwfMxw9AMWEAQBGxZMkSQ1KOHx8fH4t1fHx8jAEDBpi/d+nSxWjUqFGO25k9e7YhyYiPj7co/+WXXwxJRlhYmEX5jz/+aEgyJk+ebBiGYVy8eNFwcHAw+vbta1EvNjbWkGS0adPGXLZ582ZDktG6det77n9aWppx+/Zt46mnnjJ69OhhLo+PjzckGQ0bNjTu3LljLp8/f74hyXj22Wct2hk9erQhybhy5YphGIbx73//25Bk7Nu3754x/Fluj4dhGMbUqVMNScbvv/+eY5s7duwwJBmTJk3KtCxj/zM+6enpWbaRnp5u3L5920hISDAkGf/5z3/MywYMGJDpGvlzfBn++te/GpKMy5cvZxtrbq6ljGvW+lrKTawZMc2aNctinbCwMKN06dLZ7r9hGMalS5eM0qVLW1wnhmEYP/zwQ6ZrMOP6WbJkibmsTJkyxujRo3Pct86dO2d5LDPaq1GjhpGamprlsj9vK2M/x4wZY1H3X//6lyHJ+PTTT81lkoypU6dm2qb1ff7FF18YkozNmzdnqmt9Daxfvz7L47xy5UpDkrF48WKL7ZQuXdpISEgwl928edMoX768MWzYMHNZ/fr1je7du2faNoDihaF3AIqcf/zjH4qLi8v0eeKJJ+65brNmzfTTTz8pLCxM3377rZKTk3O93c2bN0tSpln0mjVrJn9/f3333XeSpB07dujWrVvq06ePRb0WLVpkO9tWr169sixftGiRmjRpotKlS8vOzk729vb67rvv9Msvv2Sq26lTJ5Uo8X//2fb395ekTA/OZ5SfOnVKktSoUSOVKlVKQ4cO1bJly3TixIksY7GW2+NRUCpUqCB7e3vzZ9WqVeZlSUlJGj58uLy9vc3HycfHR5KyPFb38vjjj0uS+vTpo88//1xnz57NVCe/11JeY3322Wctvjdo0EApKSmZhk/+WWxsrFJSUvT8889blLds2dK8rZw0a9ZMS5cu1fTp07Vjxw7dvn07N7uWKe689ARZx9qnTx/Z2dmZr7PCktFLaH0dP/fcc3J2ds50HTdq1EhVq1Y1fy9durT8/PyUkJBgLmvWrJnWrVunSZMmacuWLbp582bh7QAAmyFRAlDk+Pv7KzAwMNOnbNmy91w3PDxcf/3rX7Vjxw4988wzqlChgp566qlspxz/s4xhM1nNYObl5WVenvG/Hh4emeplVZZdm3PnztUrr7yi5s2ba9WqVdqxY4fi4uL09NNPZ/nDq3z58hbfS5UqlWN5SkqKpLtDkjZu3Ch3d3eNGDFCNWrUUI0aNSyetclKbo9HXnh7e0uSxY/ODFu2bFFcXJwWLVpkUZ6enq4OHTpo9erVmjBhgr777jvt3LlTO3bskKR8/Uht3bq1vvzyS6WlpSk0NFRVqlRR/fr1LZ4xyc+1lJ9YK1SoYPE9YxhYTvuVcew9PT0zLcuqzNrKlSs1YMAAffTRRwoKClL58uUVGhqqxMTEe66bIa8z/VnHZWdnpwoVKhT6cLULFy7Izs4u06QbJpNJnp6embZvfT6ku+fkz+fj/fff18SJE/Xll1+qbdu2Kl++vLp3765jx44Vzk4AsAkSJQDFip2dncaOHas9e/bo4sWLWrFihU6fPq2OHTvec+KCjB9I58+fz7Ts3Llz5ueTMur99ttvmepl90Mzq0kLPv30UwUHB2vhwoXq3LmzmjdvrsDAQF29ejXnncyHJ598Uv/973915coV7dixQ0FBQRo9erQ+++yzbNfJ7fHICy8vL9WrV08xMTHmRC5Do0aNFBgYmOn5mgMHDuinn37S7Nmz9eqrryo4OFiPP/54lj9oS5curVu3bmUqz+odPt26ddN3332nK1euaMuWLapSpYr69++v2NhYSfm7lvIS6/3IaC+r6y03yU7FihU1f/58nTx5UgkJCYqKitLq1avz9E6yrK7pnFjHlZaWpgsXLlgcGwcHhyzP3/0kUxUqVFBaWlqmiSMMw1BiYmK+rmNnZ2dNmzZNhw8fVmJiohYuXKgdO3aoa9eu+Y4TQNFDogSg2HJzc1Pv3r01YsQIXbx40TwzWXZ/sW/Xrp2kuwnMn8XFxemXX37RU089JUlq3ry5HBwctHLlSot6O3bsyLKnJDsmk8niIXJJ2r9/v/mHemEoWbKkmjdvrg8//FCStGfPnmzr5vZ45NUbb7yhP/74Q2PHjs3VBB0ZP8itj9Xf/va3THWrVaumpKQkiyQ2NTVV3377bbbtOzg4qE2bNpo5c6Ykae/evZnqZHct3U+s96NFixYqXbq0/vWvf1mUb9++PU/XoCRVrVpVI0eOVEhIiMX1YN2Lcr+sY/3888+VlpZmMWlJtWrVtH//fot6mzZt0rVr1yzKctPrliHjOrW+jletWqXr16/n+zrO4OHhoYEDB6pfv346cuTIPf8gA+Dhwax3AIqVrl27qn79+goMDFSlSpWUkJCg+fPny8fHxzwrV0BAgCTpvffe04ABA2Rvb6/atWurdu3aGjp0qD744AOVKFFCzzzzjHmWN29vb40ZM0bS3aFuY8eOVVRUlMqVK6cePXrozJkzmjZtmipXrmzxHFFOunTponfeeUdTp05VmzZtdOTIEb399tvy9fXNcta//Fq0aJE2bdqkzp07q2rVqkpJSTHPqNa+ffts18vt8cirfv366eDBg5oxY4Z++uknDRw4ULVq1VJ6erpOnz6tf/7zn5IkFxcXSVKdOnVUo0YNTZo0SYZhqHz58vrvf/+rmJiYTG337dtXb731lv7yl79o/PjxSklJ0fvvv5/p5bVvvfWWzpw5o6eeekpVqlTR5cuX9d5778ne3l5t2rSRlLtryVpeYr0f5cqV07hx4zR9+nS99NJLeu6553T69GlFRETcc+jdlStX1LZtW/Xv31916tSRi4uL4uLitH79evXs2dNcLyAgQKtXr9bChQvVtGlTlShRQoGBgfmOefXq1bKzs1NISIh51ruGDRtaPOv34osvasqUKXrrrbfUpk0bHTp0SAsWLMg07LZ+/fqSpMWLF8vFxUWlS5eWr69vlj13ISEh6tixoyZOnKjk5GS1atXKPOtd48aN9eKLL+Z5X5o3b64uXbqoQYMGKleunH755Rf985//VFBQUI6zSAJ4yNh2LgkA+D8ZM4jFxcVluTyrWbisZ8OaM2eO0bJlS6NixYpGqVKljKpVqxpDhgwxTp48abFeeHi44eXlZZQoUcJi9qw7d+4YM2fONPz8/Ax7e3ujYsWKxgsvvGCcPn3aYv309HRj+vTpRpUqVYxSpUoZDRo0ML7++mujYcOGFjORZcx698UXX2Tan1u3bhnjxo0zHnvsMaN06dJGkyZNjC+//DLTrF0ZM4nNnj3bYv3s2rY+jrGxsUaPHj0MHx8fw8HBwahQoYLRpk0b46uvvsryOP9Zbo9Hbme9+7Pvv//e6Nu3r1GlShXD3t7ecHJyMurWrWu88sorxq5duyzqHjp0yAgJCTFcXFyMcuXKGc8995xx6tSpLGdJW7t2rdGoUSPD0dHRqF69urFgwYJMs959/fXXxjPPPGM89thjRqlSpQx3d3ejU6dOxrZt28x1cnMtZTXrXW5jze6Y3WsmvQzp6elGVFSU4e3tbb4G//vf/xpt2rTJcda7lJQUY/jw4UaDBg0MV1dXw9HR0ahdu7YxdepU4/r16+b1Ll68aPTu3dtwc3MzTCaT+fhldz1mta0/7+fu3buNrl27GmXKlDFcXFyMfv36Gb/99pvF+rdu3TImTJhgeHt7G46OjkabNm2Mffv2ZbrPDePurI++vr5GyZIlLbaZ1cyHN2/eNCZOnGj4+PgY9vb2RuXKlY1XXnnFuHTpkkU9Hx8fo3Pnzpn2y/qYTpo0yQgMDDTKlStnODg4GNWrVzfGjBlj/PHHH5nWBfDwMhlGLsY9AADuKT4+XnXq1NHUqVM1efJkW4cDAADuA4kSAOTDTz/9pBUrVqhly5ZydXXVkSNHNGvWLCUnJ+vAgQPZzn4HAAAeDjyjBAD54OzsrF27dunjjz/W5cuXVbZsWQUHB2vGjBkkSQAAFAP0KAEAAACAFaYHBwAAAAArJEoAAAAAYIVECQAAAACsFPvJHNLT03Xu3Dm5uLiY35gOAAAA4NFjGIauXr0qLy+ve74gvtgnSufOnZO3t7etwwAAAABQRJw+fVpVqlTJsU6xT5RcXFwk3T0Yrq6uNo4GAAAAgK0kJyfL29vbnCPkpNgnShnD7VxdXUmUAAAAAOTqkRwmcwAAAAAAKyRKAAAAAGCFRAkAAAAArBT7Z5QAAMCjyzAMpaWl6c6dO7YOBcADULJkSdnZ2RXIa4FIlAAAQLGUmpqq8+fP68aNG7YOBcAD5OTkpMqVK6tUqVL31Q6JEgAAKHbS09MVHx+vkiVLysvLS6VKleLF80AxZxiGUlNT9fvvvys+Pl61atW650tlc0KiBAAAip3U1FSlp6fL29tbTk5Otg4HwAPi6Ogoe3t7JSQkKDU1VaVLl853W0zmAAAAiq37+WsygIdTQd33/NcDAAAAAKyQKAEAAACAFRIlAAAA5NnSpUvl5ub2QLZ14cIFubu76+TJk/fVTnBwsEaPHl0gMdnKwIED1b17d1uHUSiSkpJUqVIlnT171tahSJJMhmEYtg6iMCUnJ6ts2bK6cuWKXF1dbR0OHkLzYo6a/z0mxM+GkQCWovdFF1hbYY3C8r7S5qgC234mbcMLr208ElJSUhQfHy9fX99MD3P/+b/rD0J+/r8jMTFRUVFR+uabb3TmzBmVLVtWtWrV0gsvvKDQ0NAiMUHFzZs3dfXqVbm7uxf6tsaNG6dLly7p448/vq92Ll68KHt7e7m4uBRQZJn9/PPPGjlypHbu3Kny5ctr2LBhmjJlSoHNunjlyhUZhvHAktSCtHjxYi1fvlx79uzR1atXdenSpUz7MXbsWCUnJ+ujjz7K93Zyuv/zkhsw6x0AAEARcuLECbVq1Upubm6KjIxUQECA0tLSdPToUX3yySfy8vLSs88+a+sw5ejoKEdHx0Lfzs2bN/Xxxx9r7dq1991W+fLlCyCi7CUnJyskJERt27ZVXFycjh49qoEDB8rZ2Vmvv/56gWyjbNmyBdKOLdy4cUNPP/20nn76aYWHZ/0HsUGDBqlZs2aaPXu2ypUr94AjtMTQOwAAgCIkLCxMdnZ22rVrl/r06SN/f38FBASoV69e+uabb9S1a1dz3blz5yogIEDOzs7y9vZWWFiYrl27Zl4eERGhRo0aWbQ/f/58VatWzfx9y5YtatasmZydneXm5qZWrVopISFBkvTTTz+pbdu2cnFxkaurq5o2bapdu3ZJyjz07vjx4+rWrZs8PDxUpkwZPf7449q4caPFtqtVq6bIyEgNHjxYLi4uqlq1qhYvXpzj8Vi3bp3s7OwUFBRkEbPJZNK3336rxo0by9HRUe3atVNSUpLWrVsnf39/ubq6ql+/fhYvHLYeemcymfTll19abM/NzU1Lly7NMabs/Otf/1JKSoqWLl2q+vXrq2fPnpo8ebLmzp2rvAzi+ve//62AgAA5OjqqQoUKat++va5fvy4p89C7q1ev6vnnn5ezs7MqV66sefPmZdrPatWqafr06QoNDVWZMmXk4+Oj//znP/r999/VrVs3lSlTRgEBAeZzK90d7tivXz9VqVJFTk5OCggI0IoVK/J1XDKMHj1akyZNUosWLbKtExAQIE9PT61Zs+a+tlUQSJQAAACKiAsXLmjDhg0aMWKEnJ2ds6zz5yFcJUqU0Pvvv68DBw5o2bJl2rRpkyZMmJDr7aWlpal79+5q06aN9u/fr9jYWA0dOtS8jeeff15VqlRRXFycdu/erUmTJsne3j7Ltq5du6ZOnTpp48aN2rt3rzp27KiuXbvq1KlTFvXmzJmjwMBA7d27V2FhYXrllVd0+PDhbGP8/vvvFRgYmOWyiIgILViwQNu3b9fp06fVp08fzZ8/X8uXL9c333yjmJgYffDBB7k+HlmpV6+eypQpk+2nXr165rqxsbFq06aNHBwczGUdO3bUuXPncv181fnz59WvXz8NHjxYv/zyi7Zs2aKePXtmm2iNHTtWP/zwg7766ivFxMRo27Zt2rNnT6Z68+bNU6tWrbR371517txZL774okJDQ/XCCy9oz549qlmzpkJDQ83bSUlJUdOmTfX111/rwIEDGjp0qF588UX9+OOP5jYjIyNzPDZlypTRtm3bcrXff9asWbN8rVfQGHoHAABQRPz6668yDEO1a9e2KK9YsaJSUlIkSSNGjNDMmTMlyaLXwNfXV++8845eeeUVRUfn7hnG5ORkXblyRV26dFGNGjUkSf7+/ublp06d0vjx41WnTh1JUq1atbJtq2HDhmrYsKH5+/Tp07VmzRp99dVXGjlypLm8U6dOCgu7+1zkxIkTNW/ePG3ZssW8DWsnT56Ul5dXlsumT5+uVq1aSZKGDBmi8PBwHT9+XNWrV5ck9e7dW5s3b9bEiRPveSyys3btWt2+fTvb5X9OHBMTEy166yTJw8PDvMzX1/ee2zt//rzS0tLUs2dP+fj4SLrby5KVq1evatmyZVq+fLmeeuopSdKSJUuyPF6dOnXSsGHDJElvvfWWFi5cqMcff1zPPfecpLvnIigoSL/99ps8PT312GOPady4ceb1X331Va1fv15ffPGFmjdvLkkaPny4+vTpk+P+PPbYY/fc56zW2bt3b57XK2gkSgAAAEWM9YP/O3fuVHp6up5//nndunXLXL5582ZFRkbq0KFDSk5OVlpamlJSUnT9+vVse6T+rHz58ho4cKA6duyokJAQtW/fXn369FHlypUl3e2teOmll/TPf/5T7du313PPPWdOqKxdv35d06ZN09dff61z584pLS1NN2/ezNSj1KBBA4v99PT0VFJSUrYx3rx5M9MD+Vm15eHhIScnJ3OSlFG2c+fOex6HnGQkK7llfe4yemhyO5lDw4YN9dRTTykgIEAdO3ZUhw4d1Lt37yyf1zlx4oRu376tZs2amcvKli2bKdGWMh8ryTIByyhLSkqSp6en7ty5o3fffVcrV67U2bNndevWLd26dcviuipfvnyhPPfl6OhoMWTSVhh6BwAAUETUrFlTJpMp01C06tWrq2bNmhaTJyQkJKhTp06qX7++Vq1apd27d+vDDz+UJHMPSIkSJTIN2bLuHVmyZIliY2PVsmVLrVy5Un5+ftqxY4eku0PbDh48qM6dO2vTpk2qW7duts+OjB8/XqtWrdKMGTO0bds27du3TwEBAUpNTbWoZz10z2QyKT09PdtjUrFiRV26dCnLZX9uy2Qy5bltk8l0z+OTl6F3np6eSkxMtFg/IwnMSETupWTJkoqJidG6detUt25dffDBB6pdu7bi4+Mz1c0uCctqmJ71scquLON4zZkzR/PmzdOECRO0adMm7du3Tx07drQ4n4U19O7ixYuqVKlSntcraPQoAQAAFBEVKlRQSEiIFixYoFdffTXHXqFdu3YpLS1Nc+bMUYkSd//2/fnnn1vUqVSpkhITE2UYhvmH8L59+zK11bhxYzVu3Fjh4eEKCgrS8uXLzQ/c+/n5yc/PT2PGjFG/fv20ZMkS9ejRI1Mb27Zt08CBA83Lrl27dt/vPcqI7dNPP73vdrJSqVIlnT9/3vz92LFjmXoy8jL0LigoSJMnT1ZqaqpKlSolSdqwYYO8vLwyDcnLiclkUqtWrdSqVSu99dZb8vHx0Zo1azR27FiLejVq1JC9vb127twpb29vSXeHUx47dkxt2rTJ9faysm3bNnXr1k0vvPCCpLsJ1LFjxyyGZhbW0LsDBw4oODg4z+sVNBIlAACAIiQ6OlqtWrVSYGCgIiIi1KBBA5UoUUJxcXE6fPiwmjZtKunuj+S0tDR98MEH6tq1q3744QctWrTIoq3g4GD9/vvvmjVrlnr37q3169dr3bp15vfHxMfHa/HixXr22Wfl5eWlI0eO6OjRowoNDdXNmzc1fvx49e7dW76+vjpz5ozi4uLUq1evLOOuWbOmVq9era5du8pkMmnKlCk59ubkVseOHRUeHq5Lly4V+HTR7dq104IFC9SiRQulp6dr4sSJmXql8jL0rn///po2bZoGDhyoyZMn69ixY4qMjNRbb72V66F3P/74o7777jt16NBB7u7u+vHHH/X7779bJCgZXFxcNGDAAI0fP17ly5eXu7u7pk6dqhIlStz3e5tq1qypVatWafv27SpXrpzmzp2rxMREizjyOvQuMTFRiYmJ+vXXXyXdfedUxuyHGe3cuHFDu3fvVmRk5H3FXxBIlAAAwCOlqL88vEaNGtq7d68iIyMVHh6uM2fOyMHBQXXr1tW4cePMEyE0atRIc+fO1cyZMxUeHq7WrVsrKipKoaGh5rb8/f0VHR2tyMhIvfPOO+rVq5fGjRtnnpLbyclJhw8f1rJly3ThwgVVrlxZI0eO1LBhw5SWlqYLFy4oNDRUv/32mypWrKiePXtq2rRpWcY9b948DR48WC1btlTFihU1ceJEJScn3/fxCAgIUGBgoD7//HPzZAQFZc6cORo0aJBat24tLy8vvffee9q9e3e+2ytbtqxiYmI0YsQIBQYGqly5cho7dqxFT9DJkyfl6+urzZs3Z9lr4urqqu+//17z589XcnKyfHx8NGfOHD3zzDNZbnPu3LkaPny4unTpIldXV02YMEGnT5/O9rmu3JoyZYri4+PVsWNHOTk5aejQoerevbuuXLmS7zYXLVpkcf20bt1a0t3hnwMHDpQk/ec//1HVqlX15JNP3lf8BcFk5GVS94dQXt6+C2Tlz29wL+r/54pHS/S+3M1qlRthjcLyvtLmqALbfiZts34RIZBbKSkpio+Pl6+v733/YITtrV27VuPGjdOBAwfMwwwfVlu2bFGPHj104sSJQnmh6vXr1/XYY49pzpw5GjJkSIG3X9iaNWum0aNHq3///vluI6f7Py+5AT1KAAAAKNI6deqkY8eO6ezZs+ZncR5W69ev1+TJkwssSdq7d68OHz6sZs2a6cqVK3r77bclSd26dSuQ9h+kpKQk9e7dW/369bN1KJJIlAAAAPAQGDVqlK1DKBDvvvtugbf517/+VUeOHFGpUqXUtGlTbdu2TRUrVizw7RQ2d3f3PL0wubCRKAEAAAAPqcaNG9/Xc1XI3sM9yBMAAAAACgGJEgAAAABYIVECAAAAACskSgAAAABghUQJAAAAAKyQKAEAAACAFaYHBwAAj5bNUQ92e23DH+z2HpClS5dq9OjRunz5cqFv68KFC/L399fOnTtVrVq1fLcTHBysRo0aaf78+QUWW1FUnPfz559/1jPPPKMjR47I2dm5ULdFjxIAAEARk5iYqFGjRqlmzZoqXbq0PDw89MQTT2jRokW6ceOGrcOTJPXt21dHjx59INuKiopS165d7ytJkqTVq1frnXfeKZigspCSkqKBAwcqICBAdnZ26t69e5YxhISEqFKlSnJ1dVVQUJC+/fbbAo2jsPezsH3zzTdq3ry5HB0dVbFiRfXs2dO8LCAgQM2aNdO8efMKPQ56lAAAAIqQEydOqFWrVnJzc1NkZKQCAgKUlpamo0eP6pNPPpGXl5eeffZZW4cpR0dHOTo6Fvp2bt68qY8//lhr166977bKly9fABFl786dO3J0dNRrr72mVatWZVnn+++/V0hIiCIjI+Xm5qYlS5aoa9eu+vHHH9W4ceMCiaOw97MwrVq1Si+//LIiIyPVrl07GYahn3/+2aLOoEGDNHz4cIWHh6tkyZKFFgs9SgAAAEVIWFiY7OzstGvXLvXp00f+/v4KCAhQr1699M0336hr167munPnzlVAQICcnZ3l7e2tsLAwXbt2zbw8IiJCjRo1smh//vz5Fj0zW7ZsUbNmzeTs7Cw3Nze1atVKCQkJkqSffvpJbdu2lYuLi1xdXdW0aVPt2rVL0t2hd25ubuZ2jh8/rm7dusnDw0NlypTR448/ro0bN1psu1q1aoqMjNTgwYPl4uKiqlWravHixTkej3Xr1snOzk5BQUEWMZtMJn377bdq3LixHB0d1a5dOyUlJWndunXy9/eXq6ur+vXrZ9EDFxwcrNGjR5u/m0wmffnllxbbc3Nz09KlS3OMKTvOzs5auHChXn75ZXl6emZZZ/78+ZowYYIef/xx1apVS5GRkapVq5b++9//5mlb0dHRqlWrlrnHsXfv3uZl1vt5/vx5de7cWY6OjvL19dXy5ctVrVo1i6F5JpNJf/vb39SlSxc5OTnJ399fsbGx+vXXXxUcHCxnZ2cFBQXp+PHj5nVyc87zIi0tTaNGjdLs2bM1fPhw+fn5qXbt2hb7JkkdO3bUhQsXtHXr1nxvKzdIlAAAAIqICxcuaMOGDRoxYkS2z1+YTCbzv0uUKKH3339fBw4c0LJly7Rp0yZNmDAh19tLS0tT9+7d1aZNG+3fv1+xsbEaOnSoeRvPP/+8qlSpori4OO3evVuTJk2Svb19lm1du3ZNnTp10saNG7V371517NhRXbt21alTpyzqzZkzR4GBgdq7d6/CwsL0yiuv6PDhw9nG+P333yswMDDLZREREVqwYIG2b9+u06dPq0+fPpo/f76WL1+ub775RjExMfrggw9yfTyyUq9ePZUpUybbT7169e6r/fT0dF29ejVPvUC7du3Sa6+9prfffltHjhzR+vXr1bp162zrh4aG6ty5c9qyZYtWrVqlxYsXKykpKVO9d955R6Ghodq3b5/q1Kmj/v37a9iwYQoPDzcnyCNHjjTXz805Hz58eI7Hr0yZMub6e/bs0dmzZ1WiRAk1btxYlStX1jPPPKODBw9axFmqVCk1bNhQ27Zty/Uxyw+G3gEAABQRv/76qwzDUO3atS3KK1asqJSUFEnSiBEjNHPmTEmy6DXw9fXVO++8o1deeUXR0dG52l5ycrKuXLmiLl26qEaNGpIkf39/8/JTp05p/PjxqlOnjiSpVq1a2bbVsGFDNWzY0Px9+vTpWrNmjb766iuLH9edOnVSWFiYJGnixImaN2+etmzZYt6GtZMnT8rLyyvLZdOnT1erVq0kSUOGDFF4eLiOHz+u6tWrS5J69+6tzZs3a+LEifc8FtlZu3atbt++ne3y7BLH3JozZ46uX7+uPn365HqdU6dOydnZWV26dJGLi4t8fHyyHbZ3+PBhbdy4UXFxceaE86OPPsryXA4aNMgcx8SJExUUFKQpU6aoY8eOkqRRo0Zp0KBB5vq5Oedvv/22xo0bl+P+ZJzfEydOSLqbAM+dO1fVqlXTnDlz1KZNGx09etQimXzsscd08uTJHNu9XzbtUUpLS9Obb74pX19fOTo6qnr16nr77beVnp5urmMYhiIiIuTl5SVHR0cFBwdnyioBAACKkz/3GknSzp07tW/fPtWrV0+3bt0yl2/evFkhISF67LHH5OLiotDQUF24cEHXr1/P1XbKly+vgQMHmnsC3nvvPZ0/f968fOzYsXrppZfUvn17vfvuuxbDrqxdv35dEyZMUN26deXm5qYyZcro8OHDmXqUGjRoYLGfnp6eWfZuZLh586ZKly6d5bI/t+Xh4SEnJydzkpRRllPbueHj46OaNWtm+/Hx8cl32ytWrFBERIRWrlwpd3f3XK8XEhIiHx8fVa9eXS+++KL+9a9/ZTvJx5EjR2RnZ6cmTZqYy2rWrKly5cplqmt9PKW7kyf8uSwlJUXJycmScnfO3d3dczx+NWvWlJ3d3b6bjBzgjTfeUK9evdS0aVMtWbJEJpNJX3zxhUWsjo6OhT6xiU0TpZkzZ2rRokVasGCBfvnlF82aNUuzZ8+26CKdNWuW5s6dqwULFiguLk6enp4KCQnR1atXbRg5AABAwatZs6ZMJlOmoWjVq1dXzZo1LSZPSEhIUKdOnVS/fn2tWrVKu3fv1ocffihJ5h6QEiVKyDAMi7ase0eWLFmi2NhYtWzZUitXrpSfn5927Ngh6e5f9g8ePKjOnTtr06ZNqlu3rtasWZNl7OPHj9eqVas0Y8YMbdu2Tfv27VNAQIBSU1Mt6ln3wJhMJos/klurWLGiLl26lOWyP7dlMpny3LbJZLrn8SmsoXcrV67UkCFD9Pnnn6t9+/Z5WtfFxUV79uzRihUrVLlyZb311ltq2LBhllO1W+9fTuXWxzO7soxjmptznpehd5UrV5Yk1a1b17y+g4ODqlevninhvnjxoipVqpT9QSoANh16Fxsbq27duqlz586S7j7gt2LFCvMYSMMwNH/+fL3xxhvmaQGXLVsmDw8PLV++XMOGDbNZ7AAAAAWtQoUKCgkJ0YIFC/Tqq6/m+J6YXbt2KS0tTXPmzFGJEnf/9v35559b1KlUqZISExNlGIb5R+6+ffsytdW4cWM1btxY4eHhCgoK0vLly9WiRQtJkp+fn/z8/DRmzBj169dPS5YsUY8ePTK1sW3bNg0cONC87Nq1awUyNKpx48b69NNP77udrFSqVMmiB+3YsWOZeikKY+jdihUrNHjwYK1YscL8Oziv7Ozs1L59e7Vv315Tp06Vm5ubNm3aZDGVtiTVqVNHaWlp2rt3r5o2bSrp7hDPgnj/VW7OeV6G3jVt2lQODg46cuSInnjiCUl3E9eTJ09m6rk7cOBApkkeCppNE6WM9wEcPXpUfn5++umnn/S///3PPANHfHy8EhMT1aFDB/M6Dg4OatOmjbZv355lonTr1i2LLumMrkEAAICHQXR0tFq1aqXAwEBFRESoQYMGKlGihOLi4nT48GHzj90aNWooLS1NH3zwgbp27aoffvhBixYtsmgrODhYv//+u2bNmqXevXtr/fr1WrdunVxdXSXd/a21ePFiPfvss/Ly8tKRI0d09OhRhYaG6ubNmxo/frx69+4tX19fnTlzRnFxcerVq1eWcdesWVOrV69W165dZTKZNGXKlBx7c3KrY8eOCg8P16VLl7IcLnY/2rVrpwULFqhFixZKT0/XxIkTMyU+eR1ad+jQIaWmpurixYu6evWqOTHNmH1wxYoVCg0N1XvvvacWLVooMTFR0t2hZGXLls3VNr7++mudOHFCrVu3Vrly5bR27Vqlp6dnerZNupsotW/fXkOHDtXChQtlb2+v119/XY6OjpmGeOZVbs65u7t7rocVurq6avjw4Zo6daq8vb3l4+Oj2bNnS5Kee+45c72TJ0/q7Nmzee6JyyubJkoTJ07UlStXVKdOHZUsWVJ37tzRjBkz1K9fP0kyXzgZYyQzeHh4mKettBYVFaVp06YVbuAAAODh1Tbc1hHkqEaNGtq7d68iIyMVHh6uM2fOyMHBQXXr1tW4cePMEyE0atRIc+fO1cyZMxUeHq7WrVsrKipKoaGh5rb8/f0VHR2tyMhIvfPOO+rVq5fGjRtnnpLbyclJhw8f1rJly3ThwgVVrlxZI0eO1LBhw5SWlqYLFy4oNDRUv/32m/nFn9n9zpo3b54GDx6sli1bqmLFipo4cWKB/ME6ICBAgYGB+vzzzwt8NNGcOXM0aNAgtW7dWl5eXnrvvfe0e/fu+2qzU6dOFr9TMyZZyBjq9re//U1paWkaMWKERowYYa43YMAA87TkW7ZsUdu2bRUfH5/lS3bd3Ny0evVqRUREKCUlRbVq1dKKFSuyHQb4j3/8Q0OGDFHr1q3l6empqKgoHTx4MNtnv3KrMM757NmzZWdnpxdffFE3b95U8+bNtWnTJoskecWKFerQocN9PR+WGyYju4GLD8Bnn32m8ePHa/bs2apXr5727dun0aNHa+7cuRowYIC2b9+uVq1a6dy5c+Yxi5L08ssv6/Tp01q/fn2mNrPqUfL29taVK1fMfz0B8mJezP+9dXxMiJ8NIwEsRe/L3axWuRHWKCzvK22OKrDtZ1LEf8ii6EtJSVF8fLx8fX3v+8cgbG/t2rUaN26cDhw4YB5mWJwtXbpUM2bM0KFDh+57Vr2snDlzRt7e3tq4caOeeuqpAm+/MN26dcucGGbMeGgtp/s/OTlZZcuWzVVuYNMepfHjx2vSpEn6y1/+IunuXwwSEhIUFRWlAQMGmF/UlZiYaJEoJSUlZeplyuDg4CAHB4fCDx4AAAAPRKdOnXTs2DGdPXtW3t7etg6n0K1fv16RkZEFliRt2rRJ165dU0BAgM6fP68JEyaoWrVqOb57qahKSEjQG2+8kW2SVJBsmijduHEj018FSpYsaR7b6OvrK09PT8XExJi7LVNTU7V161bz+wMAAABQ/I0aNcrWITwwn332WYG2d/v2bU2ePFknTpyQi4uLWrZsqX/961+F0ltV2DImF3kQbJoode3aVTNmzFDVqlVVr1497d27V3PnztXgwYMl3Z2CcPTo0YqMjFStWrVUq1YtRUZGysnJSf3797dl6AAAAMBDoWPHjuaXxiL3bJooffDBB5oyZYrCwsKUlJQkLy8vDRs2TG+99Za5zoQJE3Tz5k2FhYXp0qVLat68uTZs2CAXFxcbRg4AAACgOLNpouTi4qL58+ebpwPPislkUkREhCIiIh5YXAAAAAAebcV/2hAAAAAAyCMSJQAAAACwQqIEAAAAAFZIlAAAAADAik0ncwAAAHjQovdFP9DthTUKe6Dbe1CWLl2q0aNH6/Lly4W+rQsXLsjf3187d+5UtWrV8t1OcHCwGjVqlONEYsVBcd7Pn3/+Wc8884yOHDkiZ2fnQt0WPUoAAABFTGJiokaNGqWaNWuqdOnS8vDw0BNPPKFFixbpxo0btg5PktS3b18dPXr0gWwrKipKXbt2va8kSZJWr16td955p2CCykJKSooGDhyogIAA2dnZqXv37lnW27p1q5o2barSpUurevXqWrRoUYHGUdj7WZiqVasmk8lk8Zk0aZJ5eUBAgJo1a6Z58+YVeiz0KAEAABQhJ06cUKtWreTm5qbIyEgFBAQoLS1NR48e1SeffCIvLy89++yztg5Tjo6OcnR0LPTt3Lx5Ux9//LHWrl17322VL1++ACLK3p07d+To6KjXXntNq1atyrJOfHy8OnXqpJdfflmffvqpfvjhB4WFhalSpUrq1atXgcRR2PtZ2N5++229/PLL5u9lypSxWD5o0CANHz5c4eHhKlmyZKHFQY8SAABAERIWFiY7Ozvt2rVLffr0kb+/vwICAtSrVy9988036tq1q7nu3LlzFRAQIGdnZ3l7eyssLEzXrl0zL4+IiFCjRo0s2p8/f75Fz8yWLVvUrFkzOTs7y83NTa1atVJCQoIk6aefflLbtm3l4uIiV1dXNW3aVLt27ZJ0d+idm5ubuZ3jx4+rW7du8vDwUJkyZfT4449r48aNFtuuVq2aIiMjNXjwYLm4uKhq1apavHhxjsdj3bp1srOzU1BQkEXMJpNJ3377rRo3bixHR0e1a9dOSUlJWrdunfz9/eXq6qp+/fpZ9MAFBwdr9OjR5u8mk0lffvmlxfbc3Ny0dOnSHGPKjrOzsxYuXKiXX35Znp6eWdZZtGiRqlatqvnz58vf318vvfSSBg8erL/+9a952lZ0dLRq1apl7nHs3bu3eZn1fp4/f16dO3eWo6OjfH19tXz5clWrVs1iaJ7JZNLf/vY3denSRU5OTvL391dsbKx+/fVXBQcHy9nZWUFBQTp+/Lh5ndyc8/xwcXGRp6en+WOdKHXs2FEXLlzQ1q1b73tbOSFRAgAAKCIuXLigDRs2aMSIEdk+f2Eymcz/LlGihN5//30dOHBAy5Yt06ZNmzRhwoRcby8tLU3du3dXmzZttH//fsXGxmro0KHmbTz//POqUqWK4uLitHv3bk2aNEn29vZZtnXt2jV16tRJGzdu1N69e9WxY0d17dpVp06dsqg3Z84cBQYGau/evQoLC9Mrr7yiw4cPZxvj999/r8DAwCyXRUREaMGCBdq+fbtOnz6tPn36aP78+Vq+fLm++eYbxcTE6IMPPsj18chKvXr1VKZMmWw/9erVy1N7sbGx6tChg0VZx44dtWvXLt2+fTtXbezatUuvvfaa3n77bR05ckTr169X69ats60fGhqqc+fOacuWLVq1apUWL16spKSkTPXeeecdhYaGat++fapTp4769++vYcOGKTw83Jwgjxw50lw/N+d8+PDhOR6/MmXKZLpGZs6cqQoVKqhRo0aaMWOGUlNTLZaXKlVKDRs21LZt23J1vPKLoXcAAABFxK+//irDMFS7dm2L8ooVKyolJUWSNGLECM2cOVOSLHoNfH199c477+iVV15RdHTuJqxITk7WlStX1KVLF9WoUUOS5O/vb15+6tQpjR8/XnXq1JEk1apVK9u2GjZsqIYNG5q/T58+XWvWrNFXX31l8eO6U6dOCgu7O8HFxIkTNW/ePG3ZssW8DWsnT56Ul5dXlsumT5+uVq1aSZKGDBmi8PBwHT9+XNWrV5ck9e7dW5s3b9bEiRPveSyys3bt2hwTmOwSx+wkJibKw8PDoszDw0NpaWn6448/VLly5Xu2cerUKTk7O6tLly5ycXGRj4+PGjdunGXdw4cPa+PGjYqLizMnnB999FGW53LQoEHq06ePpLvnJigoSFOmTFHHjh0lSaNGjdKgQYPM9XNzzt9++22NGzcux/358/kdNWqUmjRponLlymnnzp0KDw9XfHy8PvroI4t1HnvsMZ08eTLHdu8XiRIAAEAR8+deI0nauXOn0tPT9fzzz+vWrVvm8s2bNysyMlKHDh1ScnKy0tLSlJKSouvXr+dqRrDy5ctr4MCB6tixo0JCQtS+fXv16dPH/GN97Nixeumll/TPf/5T7du313PPPWdOqKxdv35d06ZN09dff61z584pLS1NN2/ezNRb0KBBA4v99PT0zLJ3I8PNmzdVunTpLJf9uS0PDw85OTmZk6SMsp07d97zOOTEx8fnvtbPivX5NQwjy/LshISEyMfHR9WrV9fTTz+tp59+Wj169JCTk1OmukeOHJGdnZ2aNGliLqtZs6bKlSuXqa718ZTuTp7w57KUlBQlJyfL1dU1V+fc3d1d7u7uudovSRozZoxFPOXKlVPv3r3NvUwZHB0dC31iE4beAQAAFBE1a9aUyWTKNBStevXqqlmzpsXkCQkJCerUqZPq16+vVatWaffu3frwww8lydwDUqJECfOP8AzWvSNLlixRbGysWrZsqZUrV8rPz087duyQdHdo28GDB9W5c2dt2rRJdevW1Zo1a7KMffz48Vq1apVmzJihbdu2ad++fQoICMg0bMq6B8ZkMik9PT3bY1KxYkVdunQpy2V/bstkMuW5bZPJdM/jU9BD7zw9PZWYmGhRlpSUJDs7O4tEICcuLi7as2ePVqxYocqVK+utt95Sw4YNs5yq3Xr/ciq3Pp7ZlWUc09yc8/wMvfuzFi1aSLrb2/pnFy9eVKVKlbJdryDQowQAAFBEVKhQQSEhIVqwYIFeffXVHHuFdu3apbS0NM2ZM0clStz92/fnn39uUadSpUpKTEyUYRjmH7n79u3L1Fbjxo3VuHFjhYeHKygoSMuXLzf/QPXz85Ofn5/GjBmjfv36acmSJerRo0emNrZt26aBAweal127dq1AhkY1btxYn3766X23k5VKlSrp/Pnz5u/Hjh3L1EtR0EPvgoKC9N///teibMOGDQoMDMxTW3Z2dmrfvr3at2+vqVOnys3NTZs2bVLPnj0t6tWpU0dpaWnau3evmjZtKulu0lEQ77/KzTnP69A7a3v37pWkTEMSDxw4YDGBRWEgUQIAAChCoqOj1apVKwUGBioiIkINGjRQiRIlFBcXp8OHD5t/7NaoUUNpaWn64IMP1LVrV/3www+Z3scTHBys33//XbNmzVLv3r21fv16rVu3Tq6urpLuTlW9ePFiPfvss/Ly8tKRI0d09OhRhYaG6ubNmxo/frx69+4tX19fnTlzRnFxcdlOYV2zZk2tXr1aXbt2lclk0pQpU3Lszcmtjh07Kjw8XJcuXcpyuNj9aNeunRYsWKAWLVooPT1dEydOzJSs5HXo3aFDh5SamqqLFy/q6tWr5sQ0Y/bB4cOHa8GCBRo7dqxefvllxcbG6uOPP9aKFStyvY2vv/5aJ06cUOvWrVWuXDmtXbtW6enpmZ5tk+4mSu3bt9fQoUO1cOFC2dvb6/XXX5ejo2Ouh/plJzfnPC9D72JjY7Vjxw61bdtWZcuWVVxcnMaMGaNnn31WVatWNdc7efKkzp49q/bt299X/PdCogQAAB4pYY3CbB1CjmrUqKG9e/cqMjJS4eHhOnPmjBwcHFS3bl2NGzfOPBFCo0aNNHfuXM2cOVPh4eFq3bq1oqKiFBoaam7L399f0dHRioyM1DvvvKNevXpp3Lhx5im5nZycdPjwYS1btkwXLlxQ5cqVNXLkSA0bNkxpaWm6cOGCQkND9dtvv6lixYrq2bOnpk2blmXc8+bN0+DBg9WyZUtVrFhREydOVHJy8n0fj4CAAAUGBurzzz/XsGHD7ru9P5szZ44GDRqk1q1by8vLS++995527959X2126tTJPL26JPMkCxlD3Xx9fbV27VqNGTNGH374oby8vPT+++9bJKBbtmxR27ZtFR8fn+VLdt3c3LR69WpFREQoJSVFtWrV0ooVK7IdBviPf/xDQ4YMUevWreXp6amoqCgdPHgw22e/cqugz7mDg4NWrlypadOm6datW/Lx8dHLL7+caSbHFStWqEOHDoXy/NifmYzsBi4WE8nJySpbtqyuXLli/usJkBfzYv7vreNjQvxsGAlgKXpf7ma1yo18/XDcHFVg28+kbXjhtY1HQkpKiuLj4+Xr63vfPwZhe2vXrtW4ceN04MAB8zDD4mzp0qWaMWOGDh06lOehfblx5swZeXt7a+PGjXrqqacKvP3CdOvWLXNimDHjobWc7v+85Ab0KAEAAKBI69Spk44dO6azZ8/K29vb1uEUuvXr1ysyMrLAkqRNmzbp2rVrCggI0Pnz5zVhwgRVq1Ytx3cvFVUJCQl64403sk2SChKJEgAAAIq8UaNG2TqEB+azzz4r0PZu376tyZMn68SJE3JxcVHLli31r3/9q1B6qwpbxuQiDwKJEgAAAFCMdezY0fzSWORe8R/kCQAAAAB5RKIEAACKrWI+ZxWALBTUfU+iBAAAip2MZy+sXx4KoPjLuO/v9xksnlECAADFTsmSJeXm5qakpCRJd98XdL8v1wRQtBmGoRs3bigpKUlubm4qWbLkfbVHooRH3p/fkyTl/K6kvNRF8VNQ7y3K9p1FeX0v0eX9uavn++Q9q+Rr37LYfphbg7y3k5XCekcT72d6pHh6ekqSOVkC8Ghwc3Mz3//3g0QJAAAUSyaTSZUrV5a7u7tu375t63AAPAD29vb33ZOUgUQJAAAUayVLliywH04AHh1M5gAAAAAAVkiUAAAAAMAKiRIAAAAAWCFRAgAAAAArJEoAAAAAYIVECQAAAACskCgBAAAAgBUSJQAAAACwQqIEAAAAAFZIlAAAAADAip2tAwCKmnkxR20dAoq56H3RWS+4vL9wNhi/rXDaBQCgGKNHCQAAAACskCgBAAAAgBWG3gGFxHoI35gQPxtFAjzaov88pDG7YY+5ENYorACiyaPNUYXTbtvwwmkXAIoRepQAAAAAwAqJEgAAAABYIVECAAAAACskSgAAAABghUQJAAAAAKyQKAEAAACAFaYHBx4QpgsHHm7R9zG1uDWbTDX+Z0w7DgD3RI8SAAAAAFghUQIAAAAAKwy9A+4Dw+kAS9GX9xdIO2FuDQqknUzitxVOu75PFk67AACboUcJAAAAAKyQKAEAAACAFRIlAAAAALBCogQAAAAAVkiUAAAAAMCKzROls2fP6oUXXlCFChXk5OSkRo0aaffu3eblhmEoIiJCXl5ecnR0VHBwsA4ePGjDiAEAAAAUdzZNlC5duqRWrVrJ3t5e69at06FDhzRnzhy5ubmZ68yaNUtz587VggULFBcXJ09PT4WEhOjq1au2CxwAAABAsWbT9yjNnDlT3t7eWrJkibmsWrVq5n8bhqH58+frjTfeUM+ePSVJy5Ytk4eHh5YvX65hw4Y96JABAAAAPAJs2qP01VdfKTAwUM8995zc3d3VuHFj/f3vfzcvj4+PV2Jiojp06GAuc3BwUJs2bbR9+/Ys27x165aSk5MtPgAAAACQFzbtUTpx4oQWLlyosWPHavLkydq5c6dee+01OTg4KDQ0VImJiZIkDw8Pi/U8PDyUkJCQZZtRUVGaNm1aoccOZGVezFFbhwAUC9GX99s6hLyJ35a3+peuFE4cAIACY9MepfT0dDVp0kSRkZFq3Lixhg0bppdfflkLFy60qGcymSy+G4aRqSxDeHi4rly5Yv6cPn260OIHAAAAUDzZNFGqXLmy6tata1Hm7++vU6dOSZI8PT0lydyzlCEpKSlTL1MGBwcHubq6WnwAAAAAIC9smii1atVKR44csSg7evSofHx8JEm+vr7y9PRUTEyMeXlqaqq2bt2qli1bPtBYAQAAADw6bPqM0pgxY9SyZUtFRkaqT58+2rlzpxYvXqzFixdLujvkbvTo0YqMjFStWrVUq1YtRUZGysnJSf3797dl6AAAAACKMZsmSo8//rjWrFmj8PBwvf322/L19dX8+fP1/PPPm+tMmDBBN2/eVFhYmC5duqTmzZtrw4YNcnFxsWHkAAAAAIqzAkmUkpOTtWnTJtWuXVv+/v55WrdLly7q0qVLtstNJpMiIiIUERFxn1ECAAAAQO7k6xmlPn36aMGCBZKkmzdvKjAwUH369FGDBg20atWqAg0QAAAAAB60fCVK33//vZ588klJ0po1a2QYhi5fvqz3339f06dPL9AAAQAAAOBBy1eidOXKFZUvX16StH79evXq1UtOTk7q3Lmzjh07VqABAgAAAMCDlq9EydvbW7Gxsbp+/brWr1+vDh06SJIuXbqk0qVLF2iAAAAAAPCg5Wsyh9GjR+v5559XmTJlVLVqVQUHB0u6OyQvICCgIOMDAAAAgAcuX4lSWFiYmjVrptOnTyskJEQlStztmKpevTrPKAEAAAB46OV7evDAwEA1aNBA8fHxqlGjhuzs7NS5c+eCjA0AAAAAbCJfzyjduHFDQ4YMkZOTk+rVq6dTp05Jkl577TW9++67BRogAAAAADxo+UqUwsPD9dNPP2nLli0Wkze0b99eK1euLLDgAAAAAMAW8jX07ssvv9TKlSvVokULmUwmc3ndunV1/PjxAgsOAAAAAGwhXz1Kv//+u9zd3TOVX79+3SJxAgAAAICHUb4Spccff1zffPON+XtGcvT3v/9dQUFBBRMZAAAAANhIvobeRUVF6emnn9ahQ4eUlpam9957TwcPHlRsbKy2bt1a0DECAAAAwAOVrx6lli1b6ocfftCNGzdUo0YNbdiwQR4eHoqNjVXTpk0LOkYAAAAAeKDy/R6lgIAALVu2rCBjAQAAAIAiIdeJUnJyslxdXc3/zklGPQAAAAB4GOU6USpXrpzOnz8vd3d3ubm5ZTm7nWEYMplMunPnToEGCQAAAAAPUq4TpU2bNql8+fKSpM2bNxdaQAAAAABga7lOlNq0aZPlvwEAAACguMnXrHdLlizRF198kan8iy++YIIHAAAAAA+9fCVK7777ripWrJip3N3dXZGRkfcdFAAAAADYUr4SpYSEBPn6+mYq9/Hx0alTp+47KAAAAACwpXwlSu7u7tq/f3+m8p9++kkVKlS476AAAAAAwJby9cLZv/zlL3rttdfk4uKi1q1bS5K2bt2qUaNG6S9/+UuBBggUhHkxRy2+jwnxs1EkyK3ofdH330j8NoW5Nbj/djJc/v9/IPJ9suDaBAAARVK+EqXp06crISFBTz31lOzs7jaRnp6u0NBQnlECAAAA8NDLV6JUqlQprVy5Uu+8845++uknOTo6KiAgQD4+PgUdHwAAAAA8cPlKlDL4+fnJz48hTAAAAACKl3wlSnfu3NHSpUv13XffKSkpSenp6RbLN23aVCDBAQAAAIAt5CtRGjVqlJYuXarOnTurfv36MplMBR0XAAAAANhMvhKlzz77TJ9//rk6depU0PEAAAAAgM3l6z1KpUqVUs2aNQs6FgAAAAAoEvKVKL3++ut67733ZBhGQccDAAAAADaXr6F3//vf/7R582atW7dO9erVk729vcXy1atXF0hwAAAAAGAL+UqU3Nzc1KNHj4KOBQAAAACKhHwlSkuWLCnoOAAAAACgyMjXM0qSlJaWpo0bN+pvf/ubrl69Kkk6d+6crl27VmDBAQAAAIAt5KtHKSEhQU8//bROnTqlW7duKSQkRC4uLpo1a5ZSUlK0aNGigo4TAAAAAB6YfPUojRo1SoGBgbp06ZIcHR3N5T169NB3331XYMEBAAAAgC3ke9a7H374QaVKlbIo9/Hx0dmzZwskMAAAAACwlXwlSunp6bpz506m8jNnzsjFxeW+gwIeBfNijpr/PSbEL9tlWS0H8HCLvry/QNoJc2tQIO0UVDxa06/AYrLQNrzg25SkzVGF025hxQvggcrX0LuQkBDNnz/f/N1kMunatWuaOnWqOnXqVFCxAQAAAIBN5KtHad68eWrbtq3q1q2rlJQU9e/fX8eOHVPFihW1YsWKgo4RAAAAAB6ofCVKXl5e2rdvn1asWKE9e/YoPT1dQ4YM0fPPP28xuQMAAAAAPIzylShJkqOjowYPHqzBgwcXZDwAAAAAYHP5SpT+8Y9/5Lg8NDQ0X8EAAAAAQFGQr0Rp1KhRFt9v376tGzduqFSpUnJyciJRAgAAAPBQy9esd5cuXbL4XLt2TUeOHNETTzzBZA4AAAAAHnr5SpSyUqtWLb377ruZepsAAAAA4GFTYImSJJUsWVLnzp0ryCYBAAAA4IHL1zNKX331lcV3wzB0/vx5LViwQK1atSqQwAAAAADAVvKVKHXv3t3iu8lkUqVKldSuXTvNmTOnIOICAAAAAJvJV6KUnp5e0HEAAAAAQJFRoM8oAQAAAEBxkK8epbFjx+a67ty5c/OzCQAAAACwmXwlSnv37tWePXuUlpam2rVrS5KOHj2qkiVLqkmTJuZ6JpOpYKIEAAAAgAcoX0PvunbtqjZt2ujMmTPas2eP9uzZo9OnT6tt27bq0qWLNm/erM2bN2vTpk25bjMqKkomk0mjR482lxmGoYiICHl5ecnR0VHBwcE6ePBgfkIGAAAAgFzLV4/SnDlztGHDBpUrV85cVq5cOU2fPl0dOnTQ66+/nqf24uLitHjxYjVo0MCifNasWZo7d66WLl0qPz8/TZ8+XSEhITpy5IhcXFzyEzogSZoXc9TWIeABib68v+Abjd9W8G0CAIAiJV89SsnJyfrtt98ylSclJenq1at5auvatWt6/vnn9fe//90i8TIMQ/Pnz9cbb7yhnj17qn79+lq2bJlu3Lih5cuX5ydsAAAAAMiVfCVKPXr00KBBg/Tvf/9bZ86c0ZkzZ/Tvf/9bQ4YMUc+ePfPU1ogRI9S5c2e1b9/eojw+Pl6JiYnq0KGDuczBwUFt2rTR9u3bs23v1q1bSk5OtvgAAAAAQF7ka+jdokWLNG7cOL3wwgu6ffv23Ybs7DRkyBDNnj071+189tln2rNnj+Li4jItS0xMlCR5eHhYlHt4eCghISHbNqOiojRt2rRcxwAAAAAA1vLVo+Tk5KTo6GhduHDBPAPexYsXFR0dLWdn51y1cfr0aY0aNUqffvqpSpcunW0965nzDMPIcTa98PBwXblyxfw5ffp07nYKAAAAAP6/+3rh7Pnz53X+/Hn5+fnJ2dlZhmHket3du3crKSlJTZs2lZ2dnezs7LR161a9//77srOzM/ckZfQsZUhKSsrUy/RnDg4OcnV1tfgAAAAAQF7ka+jdhQsX1KdPH23evFkmk0nHjh1T9erV9dJLL8nNzU1z5sy5ZxtPPfWUfv75Z4uyQYMGqU6dOpo4caKqV68uT09PxcTEqHHjxpKk1NRUbd26VTNnzsxP2ECRxSx8APKjUGZ1LIo2R9k6AgCPoHz1KI0ZM0b29vY6deqUnJyczOV9+/bV+vXrc9WGi4uL6tevb/FxdnZWhQoVVL9+ffM7lSIjI7VmzRodOHBAAwcOlJOTk/r375+fsAEAAAAgV/LVo7RhwwZ9++23qlKlikV5rVq1cpxoIa8mTJigmzdvKiwsTJcuXVLz5s21YcMG3qEEAAAAoFDlK1G6fv26RU9Shj/++EMODg75DmbLli0W300mkyIiIhQREZHvNgEAAAAgr/I19K5169b6xz/+Yf5uMpmUnp6u2bNnq23btgUWHAAAAADYQr56lGbPnq3g4GDt2rVLqampmjBhgg4ePKiLFy/qhx9+KOgYAQAAAOCBylePUt26dbV//341a9ZMISEhun79unr27Km9e/eqRo0aBR0jAAAAADxQee5Run37tjp06KC//e1vmjZtWmHEBAAAAAA2leceJXt7ex04cEAmk6kw4gEAAAAAm8vX0LvQ0FB9/PHHBR0LAAAAABQJ+ZrMITU1VR999JFiYmIUGBgoZ2dni+Vz584tkOAAAAAAwBbylCidOHFC1apV04EDB9SkSRNJ0tGjRy3qMCQPAAAAwMMuT4lSrVq1dP78eW3evFmS1LdvX73//vvy8PAolOAAAAAAwBbylCgZhmHxfd26dbp+/XqBBgQUhHkxR+9dCYUiel903leK31bwgQAAANyHfE3mkME6cQIAAACA4iBPiZLJZMr0DBLPJAEAAAAobvI89G7gwIFycHCQJKWkpGj48OGZZr1bvXp1wUUIAAAAAA9YnhKlAQMGWHx/4YUXCjQYAAAAACgK8pQoLVmypLDiAAAAAIAi474mcwAAAACA4ohECQAAAACskCgBAAAAgBUSJQAAAACwQqIEAAAAAFZIlAAAAADACokSAAAAAFghUQIAAAAAKyRKAAAAAGCFRAkAAAAArJAoAQAAAIAVEiUAAAAAsEKiBAAAAABWSJQAAAAAwAqJEgAAAABYIVECAAAAACskSgAAAABghUQJAAAAAKyQKAEAAACAFRIlAAAAALBCogQAAAAAVuxsHQBQUObFHLV1CA+t6H3RuasYv61wAwEAACgi6FECAAAAACskSgAAAABghUQJAAAAAKyQKAEAAACAFRIlAAAAALDCrHfAQ8Z6dr8xIX42igQACk/05f0F0k6YW4MCaadI2BxVOO22DS+cdoGHHD1KAAAAAGCFRAkAAAAArJAoAQAAAIAVEiUAAAAAsEKiBAAAAABWSJQAAAAAwAqJEgAAAABYIVECAAAAACskSgAAAABgxc7WAQC4t3kxR20dAgDkSvTl/bYOwfY2R9k6AgAFgB4lAAAAALBCogQAAAAAVmyaKEVFRenxxx+Xi4uL3N3d1b17dx05csSijmEYioiIkJeXlxwdHRUcHKyDBw/aKGIAAAAAjwKbPqO0detWjRgxQo8//rjS0tL0xhtvqEOHDjp06JCcnZ0lSbNmzdLcuXO1dOlS+fn5afr06QoJCdGRI0fk4uJiy/BhYzy3I+1JXqnofRUyL4jf9uCDAQAAKEZsmiitX7/e4vuSJUvk7u6u3bt3q3Xr1jIMQ/Pnz9cbb7yhnj17SpKWLVsmDw8PLV++XMOGDbNF2AAAAACKuSL1jNKVK1ckSeXLl5ckxcfHKzExUR06dDDXcXBwUJs2bbR9+/Ys27h165aSk5MtPgAAAACQF0UmUTIMQ2PHjtUTTzyh+vXrS5ISExMlSR4eHhZ1PTw8zMusRUVFqWzZsuaPt7d34QYOAAAAoNgpMonSyJEjtX//fq1YsSLTMpPJZPHdMIxMZRnCw8N15coV8+f06dOFEi8AAACA4qtIvHD21Vdf1VdffaXvv/9eVapUMZd7enpKutuzVLlyZXN5UlJSpl6mDA4ODnJwcCjcgAEAAAAUazbtUTIMQyNHjtTq1au1adMm+fr6Wiz39fWVp6enYmJizGWpqanaunWrWrZs+aDDBQAAAPCIsGmP0ogRI7R8+XL95z//kYuLi/m5o7Jly8rR0VEmk0mjR49WZGSkatWqpVq1aikyMlJOTk7q37+/LUMHAAAAUIzZNFFauHChJCk4ONiifMmSJRo4cKAkacKECbp586bCwsJ06dIlNW/eXBs2bOAdSgAAAAAKjU0TJcMw7lnHZDIpIiJCERERhR8QAAAAAKgIzXoHAAAAAEUFiRIAAAAAWCFRAgAAAAArJEoAAAAAYIVECQAAAACskCgBAAAAgBUSJQAAAACwQqIEAAAAAFZIlAAAAADAip2tAwBya17MUVuHUGD2JK+UJFVJ3n1f7VSRpHhHi7LTl2+a/+3tZrkMAJB/0Zf3F0g7YW4NCqSdIm9zVOG02za8cNoFrNCjBAAAAABWSJQAAAAAwAqJEgAAAABY4RklFGlF5bmkFqcW66sSvxZYe1UKrCUAAAAUBnqUAAAAAMAKiRIAAAAAWCFRAgAAAAArJEoAAAAAYIVECQAAAACskCgBAAAAgBUSJQAAAACwQqIEAAAAAFZ44SzwkDt9+aatQwCAIiv68n5bh1D0bY6ydQRAkUSPEgAAAABYIVECAAAAACskSgAAAABghWeUUKTMizl6X+u3OLU4U9lXJX69rzbvtnHfTQAAAOAhws8/AAAAALBCogQAAAAAVkiUAAAAAMAKiRIAAAAAWCFRAgAAAAArJEoAAAAAYIVECQAAAACskCgBAAAAgBUSJQAAAACwYmfrAPCI2hwlSYo9ccGiuMWf/v1ViV/z3OxXpP4AAAAoAPysBAAAAAArJEoAAAAAYIWhd8jZ/x8il1vRl/ffs87pyzf/7wupOgAAAIogfqYCAAAAgBUSJQAAAACwQqIEAAAAAFZIlAAAAADACokSAAAAAFghUQIAAAAAK0wPDjxCLKZml+Tt5mijSAAA9ys3r+TIrTC3BgXW1kMrj69EybW24YXTLgodPUoAAAAAYIVECQAAAACsMPTuQSvgbt2C7HYvKNbDu/BwYFgeAOB+FdTvEoYC5gJDBQsdPUoAAAAAYIVECQAAAACsMPQOKMbuNQySYZIA8OAVxWHzRTGmbBXWkLPC8rDFCzN6lAAAAADACokSAAAAAFh5KIbeRUdHa/bs2Tp//rzq1aun+fPn68knn7R1WECxxix4AABb4WW6NsRsemZFvkdp5cqVGj16tN544w3t3btXTz75pJ555hmdOnXK1qEBAAAAKKaKfKI0d+5cDRkyRC+99JL8/f01f/58eXt7a+HChbYODQAAAEAxVaSH3qWmpmr37t2aNGmSRXmHDh20ffv2LNe5deuWbt26Zf5+5coVSVJycnLhBZoX11MKtLmbN24XaHsF4dbNohcT7t/NUkX6PxcAAGQp2b5gf3shn4rIb/GMnMAwjHvWLdK/fP744w/duXNHHh4eFuUeHh5KTEzMcp2oqChNmzYtU7m3t3ehxAgAAICia5xW2ToESJLetnUAFq5evaqyZcvmWKdIJ0oZTCaTxXfDMDKVZQgPD9fYsWPN39PT03Xx4kVVqFAh23UeNsnJyfL29tbp06fl6upq63AeWZyHooNzUTRwHooOzkXRwHkoOjgXRYetz4VhGLp69aq8vLzuWbdIJ0oVK1ZUyZIlM/UeJSUlZeplyuDg4CAHBweLMjc3t8IK0aZcXV252YsAzkPRwbkoGjgPRQfnomjgPBQdnIuiw5bn4l49SRmK9GQOpUqVUtOmTRUTE2NRHhMTo5YtW9ooKgAAAADFXZHuUZKksWPH6sUXX1RgYKCCgoK0ePFinTp1SsOHD7d1aAAAAACKqSKfKPXt21cXLlzQ22+/rfPnz6t+/fpau3atfHx8bB2azTg4OGjq1KmZhhjiweI8FB2ci6KB81B0cC6KBs5D0cG5KDoepnNhMnIzNx4AAAAAPEKK9DNKAAAAAGALJEoAAAAAYIVECQAAAACskCgBAAAAgBUSpYdctWrVZDKZLD6TJk2ydViPhOjoaPn6+qp06dJq2rSptm3bZuuQHikRERGZrn1PT09bh/VI+P7779W1a1d5eXnJZDLpyy+/tFhuGIYiIiLk5eUlR0dHBQcH6+DBg7YJtpi717kYOHBgpvukRYsWtgm2GIuKitLjjz8uFxcXubu7q3v37jpy5IhFHe6Lwpeb88A98WAsXLhQDRo0ML9UNigoSOvWrTMvf1juBxKlYiBj6vSMz5tvvmnrkIq9lStXavTo0XrjjTe0d+9ePfnkk3rmmWd06tQpW4f2SKlXr57Ftf/zzz/bOqRHwvXr19WwYUMtWLAgy+WzZs3S3LlztWDBAsXFxcnT01MhISG6evXqA460+LvXuZCkp59+2uI+Wbt27QOM8NGwdetWjRgxQjt27FBMTIzS0tLUoUMHXb9+3VyH+6Lw5eY8SNwTD0KVKlX07rvvateuXdq1a5fatWunbt26mZOhh+Z+MPBQ8/HxMebNm2frMB45zZo1M4YPH25RVqdOHWPSpEk2iujRM3XqVKNhw4a2DuORJ8lYs2aN+Xt6errh6elpvPvuu+aylJQUo2zZssaiRYtsEOGjw/pcGIZhDBgwwOjWrZtN4nmUJSUlGZKMrVu3GobBfWEr1ufBMLgnbKlcuXLGRx999FDdD/QoFQMzZ85UhQoV1KhRI82YMUOpqam2DqlYS01N1e7du9WhQweL8g4dOmj79u02iurRdOzYMXl5ecnX11d/+ctfdOLECVuH9MiLj49XYmKixf3h4OCgNm3acH/YyJYtW+Tu7i4/Pz+9/PLLSkpKsnVIxd6VK1ckSeXLl5fEfWEr1uchA/fEg3Xnzh199tlnun79uoKCgh6q+8HO1gHg/owaNUpNmjRRuXLltHPnToWHhys+Pl4fffSRrUMrtv744w/duXNHHh4eFuUeHh5KTEy0UVSPnubNm+sf//iH/Pz89Ntvv2n69Olq2bKlDh48qAoVKtg6vEdWxj2Q1f2RkJBgi5Aeac8884yee+45+fj4KD4+XlOmTFG7du20e/duOTg42Dq8YskwDI0dO1ZPPPGE6tevL4n7whayOg8S98SD9PPPPysoKEgpKSkqU6aM1qxZo7p165qToYfhfiBRKoIiIiI0bdq0HOvExcUpMDBQY8aMMZc1aNBA5cqVU+/evc29TCg8JpPJ4rthGJnKUHieeeYZ878DAgIUFBSkGjVqaNmyZRo7dqwNI4PE/VFU9O3b1/zv+vXrKzAwUD4+Pvrmm2/Us2dPG0ZWfI0cOVL79+/X//73v0zLuC8enOzOA/fEg1O7dm3t27dPly9f1qpVqzRgwABt3brVvPxhuB9IlIqgkSNH6i9/+UuOdapVq5ZlecbMLb/++iuJUiGpWLGiSpYsman3KCkpKdNfR/DgODs7KyAgQMeOHbN1KI+0jJkHExMTVblyZXM590fRULlyZfn4+HCfFJJXX31VX331lb7//ntVqVLFXM598WBldx6ywj1ReEqVKqWaNWtKkgIDAxUXF6f33ntPEydOlPRw3A88o1QEVaxYUXXq1MnxU7p06SzX3bt3ryRZXHgoWKVKlVLTpk0VExNjUR4TE6OWLVvaKCrcunVLv/zyC9e+jfn6+srT09Pi/khNTdXWrVu5P4qACxcu6PTp09wnBcwwDI0cOVKrV6/Wpk2b5Ovra7Gc++LBuNd5yAr3xINjGIZu3br1UN0P9Cg9xGJjY7Vjxw61bdtWZcuWVVxcnMaMGaNnn31WVatWtXV4xdrYsWP14osvKjAwUEFBQVq8eLFOnTql4cOH2zq0R8a4cePUtWtXVa1aVUlJSZo+fbqSk5M1YMAAW4dW7F27dk2//vqr+Xt8fLz27dun8uXLq2rVqho9erQiIyNVq1Yt1apVS5GRkXJyclL//v1tGHXxlNO5KF++vCIiItSrVy9VrlxZJ0+e1OTJk1WxYkX16NHDhlEXPyNGjNDy5cv1n//8Ry4uLuYRB2XLlpWjo6NMJhP3xQNwr/Nw7do17okHZPLkyXrmmWfk7e2tq1ev6rPPPtOWLVu0fv36h+t+sNl8e7hvu3fvNpo3b26ULVvWKF26tFG7dm1j6tSpxvXr120d2iPhww8/NHx8fIxSpUoZTZo0sZh+FIWvb9++RuXKlQ17e3vDy8vL6Nmzp3Hw4EFbh/VI2Lx5syEp02fAgAGGYdydCnnq1KmGp6en4eDgYLRu3dr4+eefbRt0MZXTubhx44bRoUMHo1KlSoa9vb1RtWpVY8CAAcapU6dsHXaxk9U5kGQsWbLEXIf7ovDd6zxwTzw4gwcPNv9GqlSpkvHUU08ZGzZsMC9/WO4Hk2EYxoNMzAAAAACgqOMZJQAAAACwQqIEAAAAAFZIlAAAAADACokSAAAAAFghUQIAAAAAKyRKAAAAAGCFRAkAAAAArJAoAQAAAIAVEiUAQLETHBys0aNH2zoMAMBDjEQJAFCkdO3aVe3bt89yWWxsrEwmk/bs2fOAowIAPGpIlAAARcqQIUO0adMmJSQkZFr2ySefqFGjRmrSpIkNIgMAPEpIlAAARUqXLl3k7u6upUuXWpTfuHFDK1euVPfu3dWvXz9VqVJFTk5OCggI0IoVK3Js02Qy6csvv7Qoc3Nzs9jG2bNn1bdvX5UrV04VKlRQt27ddPLkSfPyLVu2qFmzZnJ2dpabm5tatWqVZTIHACgeSJQAAEWKnZ2dQkNDtXTpUhmGYS7/4osvlJqaqpdeeklNmzbV119/rQMHDmjo0KF68cUX9eOPP+Z7mzdu3FDbtm1VpkwZff/99/rf//6nMmXK6Omnn1ZqaqrS0tLUvXt3tWnTRvv371dsbKyGDh0qk8lUELsMACiC7GwdAAAA1gYPHqzZs2dry5Ytatu2raS7w+569uypxx57TOPGjTPXffXVV7V+/Xp98cUXat68eb6299lnn6lEiRL66KOPzMnPkiVL5Obmpi1btigwMFBXrlxRly5dVKNGDUmSv7//fe4lAKAoo0cJAFDk1KlTRy1bttQnn3wiSTp+/Li2bdumwYMH686dO5oxY4YaNGigChUqqEyZMtqwYYNOnTqV7+3t3r1bv/76q1xcXFSmTBmVKVNG5cuXV0pKio4fP67y5ctr4MCB6tixo7p27ar33ntP58+fL6jdBQAUQSRKAIAiaciQIVq1apWSk5O1ZMkS+fj46KmnntKcOXM0b948TZgwQZs2bdK+ffvUsWNHpaamZtuWyWSyGMYnSbdv3zb/Oz09XU2bNtW+ffssPkePHlX//v0l3e1hio2NVcuWLbVy5Ur5+flpx44dhbPzAACbI1ECABRJffr0UcmSJbV8+XItW7ZMgwYNkslk0rZt29StWze98MILatiwoapXr65jx47l2FalSpUseoCOHTumGzdumL83adJEx44dk7u7u2rWrGnxKVu2rLle48aNFR4eru3bt6t+/fpavnx5we84AKBIIFECABRJZcqUUd++fTV58mSdO3dOAwcOlCTVrFlTMTEx2r59u3755RcNGzZMiYmJObbVrl07LViwQHv27NGuXbs0fPhw2dvbm5c///zzqlixorp166Zt27YpPj5eW7du1ahRo3TmzBnFx8crPDxcsbGxSkhI0IYNG3T06FGeUwKAYoxECQBQZA0ZMkSXLl1S+/btVbVqVUnSlClT1KRJE3Xs2FHBwcHy9PRU9+7dc2xnzpw58vb2VuvWrdW/f3+NGzdOTk5O5uVOTk76/vvvVbVqVfXs2VP+/v4aPHiwbt68KVdXVzk5Oenw4cPq1auX/Pz8NHToUI0cOVLDhg0rzN0HANiQybAetA0AAAAAjzh6lAAAAADACokSAAAAAFghUQIAAAAAKyRKAAAAAGCFRAkAAAAArJAoAQAAAIAVEiUAAAAAsEKiBAAAAABWSJQAAAAAwAqJEgAAAABYIVECAAAAACv/D/pz/GtVOFFxAAAAAElFTkSuQmCC",
            "text/plain": [
              "<Figure size 1000x400 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "viz_distribusi_gaussian(first_gaussian, second_gaussian, third_gaussian)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5PFMTW6aDwzS"
      },
      "source": [
        "##### Output yang diharapkan\n",
        "\n",
        "<img src=\"https://drive.google.com/uc?id=1jj6RZ6AMkQvHnQM3hJ82X6ZAdJv5RPfz\" style=\"height:300px;\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ohTwnJs_FuKO"
      },
      "source": [
        "### Tugas 3: Membuat Data Binomial Distribution\n",
        "\n",
        "Selanjutnya, kita akan membuat data yang mengikuti distribusi binomial. Binomial adalah jenis distribusi untuk variabel acak diskret yang menghitung jumlah keberhasilan (success) dalam sejumlah percobaan yang sudah ditetapkan.\n",
        "\n",
        "Ada beberapa sifat utama dalam distribusi binomial, yakni berikut.\n",
        "- Jumlah percobaan tetap (n).\n",
        "- Percobaan saling independen, artinya setiap percobaan tidak memengaruhi percobaan lain.\n",
        "- Peluang sukses tetap (p) setiap percobaan.\n",
        "  - Contohnya, saat Anda mencoba melempar koin, peluang keluar kepala (head) selalu 0.5 setiap kali dilempar.\n",
        "- Hanya dua hasil pada setiap percobaan, yakni berhasil atau gagal."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CQU3ijUrJ5Nd"
      },
      "source": [
        "#### Tugas 3.1: Distribusi Binomial - Inverse CDF\n",
        "\n",
        "Jika suatu variabel acak $X$ mengikuti bentuk $Binomial(n,p)$, rumus PDF nya adalah sebagai berikut.\n",
        "\n",
        "$$P(X = k) = {n \\choose k}p^{k}(1-p)^{n-k}.$$\n",
        "\n",
        "Oleh sebab itu, jika $0 \\leq x \\leq n$, rumus CDF-nya sebagai berikut.\n",
        "\n",
        "$$F(x) = P(X \\leq x) = P(X = 0) + P(X = 1) + \\ldots + P(X = \\lfloor x \\rfloor) = \\sum_{k=0}^{\\lfloor x \\rfloor} {n \\choose k}p^{k}(1-p)^{n-k}$$\n",
        "\n",
        "Dalam rumus tersebut, notasi $\\lfloor x \\rfloor$ melambangkan floor function yang mengembalikan bilangan bulat terbesar yang kurang dari atau sama dengan x. Misalnya, $\\lfloor 4.7 \\rfloor = 4$.\n",
        "\n",
        "Fungsi ini diperlukan karena domain $F$ adalah bilangan riil, tetapi $P(X = k)$ hanya tidak nol untuk nilai-nilai bilangan bulat positif. Selain itu, jika $x > n$, maka $F(x) = 1$.\n",
        "\n",
        "Sama seperti Gaussian, binomial pun tidak memiliki bentuk tertutup (closed-form) untuk fungsi inverse CDF $F$ untuk kasus ini. Anda dapat memanfaatkan library `scipy.stats.binom` yang mengimplementasikan inverse CDF menggunakan **generalized quantile functions**.\n",
        "\n",
        "> Tips: Anda dapat menggunakan method .ppf dalam scipy.stats.binom untuk mengimplementasikan rumus inverse CDF"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "id": "Ep6BH12YFvr3"
      },
      "outputs": [],
      "source": [
        "from scipy.stats import binom\n",
        "\n",
        "def binomial_inverse_cdf(probability, n_trials, success_prob):\n",
        "  \"\"\"\n",
        "  Menghitung inverse CDF dari distribusi Binomial.\n",
        "\n",
        "  Parameters:\n",
        "  - probability (float atau ndarray): Probabilitas atau array dari probabilitas.\n",
        "  - n_trials (int): Banyaknya percobaan yang dilakukan.\n",
        "  - success_prob (float): Jumlah percobaan yang berhasil.\n",
        "\n",
        "  Returns:\n",
        "  - array (float atau ndarray): Nilai inverse CDF untuk distribusi Binomial berdasarkan probabilitas yang diberikan.\n",
        "  \"\"\"\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "  array = binom.ppf(probability, n_trials, success_prob)\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return array"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "SIxnH2uwP7VB"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Inverse CDF dari distribusi Binomial dengan probability 1e-05, n_trials 15, dan success_prob 0.9:  7.000\n",
            "Inverse CDF dari distribusi Binomial dengan probability 0, n_trials 5, dan success_prob 0.1: -1.000\n",
            "Inverse CDF dari distribusi Binomial dengan probability 0.3, n_trials 22, dan success_prob 0.5:  10.000\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "print(f\"Inverse CDF dari distribusi Binomial dengan probability {1e-5}, n_trials {15}, dan success_prob {0.9}: {binomial_inverse_cdf(1e-5, 15, 0.9): .3f}\")\n",
        "print(f\"Inverse CDF dari distribusi Binomial dengan probability {0}, n_trials {5}, dan success_prob {0.1}: {binomial_inverse_cdf(0, 5, 0.1): .3f}\")\n",
        "print(f\"Inverse CDF dari distribusi Binomial dengan probability {0.3}, n_trials {22}, dan success_prob {0.5}: {binomial_inverse_cdf(0.3, 22, 0.5): .3f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kkjE8xqSSRZA"
      },
      "source": [
        "##### Output yang diharapkan\n",
        "\n",
        "```\n",
        "Inverse CDF dari distribusi Binomial dengan probability 1e-05, n_trials 15, dan success_prob 0.9:  7.000\n",
        "Inverse CDF dari distribusi Binomial dengan probability 0, n_trials 5, dan success_prob 0.1: -1.000\n",
        "Inverse CDF dari distribusi Binomial dengan probability 0.3, n_trials 22, dan success_prob 0.5:  10.000\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yR-F9q7IP6kM"
      },
      "source": [
        "#### Tugas 3.2: Menghasilkan Binomial Distribution\n",
        "\n",
        "Mantap! Anda sudah membuat fungsi yang menghasilkan inverse CDF untuk distribusi binomial.\n",
        "\n",
        "Sekarang, mari buat fungsi yang membuat data acak dengan distribusi binomial."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "aYxihg-bTe4S"
      },
      "outputs": [],
      "source": [
        "def generate_binomial(n_trials, success_prob, num_samples):\n",
        "  \"\"\"\n",
        "  Menghasilkan sebuah array dari angka acak dengan distribusi binomial.\n",
        "\n",
        "  Parameters:\n",
        "  - n_trials (int): Banyaknya percobaan yang dilakukan untuk distribusi binomial.\n",
        "  - success_prob (float): Banyaknya percobaan yang berhasil.\n",
        "  - num_samples(int): Banyaknya sampel yang akan dihasilkan.\n",
        "\n",
        "  Returns:\n",
        "  - array (ndarray): Sebuah array yang berisi angka acak dengan distribusi Binomial\n",
        "  \"\"\"\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "\n",
        "  # Petunjuk:\n",
        "  # - Gunakan fungsi generate_rand_uniform untuk menghasilkan data acak uniform dengan rentang 0 dan 1.\n",
        "  array = binom.ppf(generate_rand_uniform(0, 1, num_samples), n_trials, success_prob)\n",
        "  # Petunjuk:\n",
        "  # - Gunakan fungsi binom_inverse_cdf untuk melakukan inverse CDF dari data acak yang dihasilkan pada fungsi sebelumnya.\n",
        "\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return array"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "id": "nKTYrTL2UP6_"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "3 angka pertama untuk variabel first_binomial: [7. 3. 2.]\n",
            "3 angka pertama untuk variabel second_binomial: [9. 6. 4.]\n",
            "3 angka pertama untuk variabel third_binomial: [15.15687921  5.40209587  1.98028072]\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "first_binomial = generate_binomial(12, 0.4, 1000)\n",
        "second_binomial = generate_binomial(15, 0.5, 1000)\n",
        "third_binomial = generate_binomial(25, 0.8, 1000)\n",
        "\n",
        "print(f\"3 angka pertama untuk variabel first_binomial: {first_binomial[:3]}\")\n",
        "print(f\"3 angka pertama untuk variabel second_binomial: {second_binomial[:3]}\")\n",
        "print(f\"3 angka pertama untuk variabel third_binomial: {third_gaussian[:3]}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "R79ZOEj9gsDk"
      },
      "source": [
        "##### Output yang diharapkan\n",
        "\n",
        "```\n",
        "3 angka pertama untuk variabel first_binomial: [7. 3. 2.]\n",
        "3 angka pertama untuk variabel second_binomial: [9. 6. 4.]\n",
        "3 angka pertama untuk variabel third_binomial: [15.15687921  5.40209587  1.98028072]\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "s4NW6exCg4Ck"
      },
      "source": [
        "#### Tugas 3.3: Membuat Visualisasi Ditribusi Binomial.\n",
        "\n",
        "*Marvelous!* Anda sudah membuat fungsi untuk menghasilkan angka acak distribusi binomial. Lanjut, mari kita buat visualisasinya.\n",
        "\n",
        "**Catatan:**\n",
        "\n",
        "Fungsi ini tidak akan masuk dalam pengujian, silakan untuk berkreasi hingga memenuhi ekspektasi yang diharapkan."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "metadata": {
        "id": "CMDEdZoZUfdg"
      },
      "outputs": [],
      "source": [
        "def viz_distribusi_binomial(first_binomial, second_binomial, third_binomial):\n",
        "  \"\"\"\n",
        "  Menghasilkan histogram plot untuk tiga distribusi binomial yang dihasilkan pada pengujian.\n",
        "\n",
        "  Parameters:\n",
        "  - first_binomial (ndarray): array dari distribusi binomial pertama.\n",
        "  - second_binomial (ndarray): array dari distribusi binomial kedua.\n",
        "  - third_binomial (ndarray): array dari distribusi binomial ketiga.\n",
        "  \"\"\"\n",
        "\n",
        "  fig, ax = plt.subplots(1, 1, figsize=(10, 4))\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "  # Tugas:\n",
        "  # Buatlah tiga buah histogram plot yang diambil dari variabel first_binomial,\n",
        "  # second_binomial, dan third_binomial dengan menggunakan parameter-parameter berikut.\n",
        "  # - alpha=0.5,\n",
        "  # - Menggunakan label sesuai dengan jenis distribusi Gaussian-nya.\n",
        "  ax.hist(first_binomial, bins=32, alpha=0.5, label='Binomial (n_trials=12, success_prob=0.4)')\n",
        "  ax.hist(second_binomial, bins=32, alpha=0.5, label='Binomial (n_trials=15, success_prob=0.5)')\n",
        "  ax.hist(third_binomial, bins=32, alpha=0.5, label='Binomial (n_trials=25, success_prob=0.8)')\n",
        "  \n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  ax.set_title(\"Histograms of Binomial distributions\")\n",
        "  ax.set_xlabel(\"Values\")\n",
        "  ax.set_ylabel(\"Frequencies\")\n",
        "  ax.legend()\n",
        "  plt.savefig(\"binomial.png\")\n",
        "  plt.savefig(\"binomial-viz.png\")\n",
        "  plt.show()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "xJBt9-t2UmxI"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1IAAAGHCAYAAAC6SmOyAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABvpUlEQVR4nO3de1xNaf8//tdWSicd1S46mApRjjmUoSIdyCmGkUFO8XGWDA2jnA2TGCbMjNMYjHuG3G433WNEoiLcOd2Nw0wIJUNKZ2r//vBrfds67bWVitfz8diPsdd1rWu912rtNfu9r2tdSyKTyWQgIiIiIiIihTWq6wCIiIiIiIgaGiZSREREREREIjGRIiIiIiIiEomJFBERERERkUhMpIiIiIiIiERiIkVERERERCQSEykiIiIiIiKRmEgRERERERGJxESKiIiIiIhIJCZSRNTg7Nq1CxKJBBcvXqyw3MfHB1ZWVnLLrKys4O/vL2o7cXFxCA0NxfPnz5ULlCq0ePFiWFhYQFVVFXp6epXWCw0NhUQiEV6NGjWCqakp+vfvj3PnzsnVvXv3LiQSCXbt2lW7wb+l0n161+sCgL+/f7nPhUQiQWhoqKh2jh07JnqdirZV3edYGY8ePUJoaCiSkpLKlb3t8SMiepNqXQdARPQuREZGomnTpqLWiYuLw9KlS+Hv71/lF35S3D//+U+sXLkSixYtgre3N9TV1atdJyoqCrq6uigpKcH9+/exdu1auLq64vz58+jcuTMAwNTUFPHx8bC2tq7tXXgrkyZNgpeXV12HIYiPj0eLFi1ErXPs2DF8++23opMpZbYl1qNHj7B06VJYWVmhY8eOcmX17dgTUcPHRIqIPgidOnWq6xBEe/nyJSQSCVRV359L9fXr1wEAs2bNgrGxsULrdOnSBUZGRgAAZ2dndOvWDdbW1vj111+FREpdXR09evSonaBrUIsWLWo9mRCjto+ZTCZDQUEBNDQ06vzvU9+OPRE1fBzaR0QfhDeH9pWUlGDFihVo3bo1NDQ0oKenh/bt22Pjxo0AXg8Dmj9/PgCgZcuWwvCy06dPC+uvXbsWbdq0gbq6OoyNjTF27Fg8ePBAbrsymQyrVq2CpaUlmjRpAkdHR5w4cQKurq5wdXUV6p0+fRoSiQR79uzBvHnz0Lx5c6irq+POnTt48uQJpk2bhrZt20JbWxvGxsbo06cPYmNj5bZVOrxt3bp1+Oqrr2BlZQUNDQ24urri1q1bePnyJRYuXAgzMzPo6upi6NChyMjIkGsjOjoarq6uMDQ0hIaGBiwsLDBs2DDk5eVVeXwVOR5WVlZYvHgxAMDExESpYWUAoKurCwBo3LhxuX0vO7SvdCjXjRs3MGrUKOjq6sLExAQTJkxAVlaWXJsFBQUIDg5Gy5YtoaamhubNm2P69OnlhnVaWVnBx8cHR48eRadOnaChoQE7OzscPXoUwOvhanZ2dtDS0kK3bt3KDVuraHjZgQMH4OHhAVNTU6G9hQsXIjc3V/SxKbVr1y60bt0a6urqsLOzw48//lhhvTf/Bnl5eQgKCkLLli3RpEkTGBgYwNHREfv37wfwenjgt99+K6xb+rp7966wbMaMGdi6dSvs7Oygrq6O3bt3V7itUpmZmRg/fjwMDAygpaWFgQMH4q+//pKrU9nQ3LKfo9OnT6Nr164AgPHjxwuxlW6zomOv6OfY1dUV9vb2SExMRK9evaCpqYmPPvoIa9asQUlJiVx7VV1XiOj98v78zElEH5zi4mK8evWq3HKZTFbtumvXrkVoaCgWL16M3r174+XLl/jjjz+EL86TJk3Cs2fPsGnTJhw6dAimpqYAgLZt2wIA/u///g/fffcdZsyYAR8fH9y9exdffvklTp8+jcuXLws9KIsWLcLq1asREBAAX19fpKamYtKkSXj58iVatWpVLq7g4GA4OTlh69ataNSoEYyNjfHkyRMAQEhICKRSKXJychAZGQlXV1ecPHlSLiEDgG+//Rbt27fHt99+i+fPn2PevHkYOHAgunfvjsaNG2PHjh24d+8egoKCMGnSJBw5cgTA62RkwIAB6NWrF3bs2AE9PT08fPgQUVFRKCoqgqamZqXHU5HjERkZiW+//Rbbt28Xhusp0kNQ+ncuHdq3ePFiqKurY/jw4dWuCwDDhg3DyJEjMXHiRFy7dg3BwcEAgB07dgB4fb4MGTIEJ0+eRHBwMHr16oWrV68iJCQE8fHxiI+PlxuCeOXKFQQHB2PRokXQ1dXF0qVL4evri+DgYJw8eRKrVq2CRCLBggUL4OPjg5SUFGhoaFQa3+3bt9G/f3/MmTMHWlpa+OOPP/DVV1/hwoULiI6OVmgfy9q1axfGjx+PwYMHIywsDFlZWQgNDUVhYSEaNar699PAwEDs2bMHK1asQKdOnZCbm4vr16/j6dOnAIAvv/wSubm5+PXXXxEfHy+sV/r5AIDDhw8jNjYWS5YsgVQqrbbnceLEiejXrx/27duH1NRULF68GK6urrh69aqoIbWdO3fGzp07MX78eCxevBgDBgwAgCrPMUU/xwCQnp6O0aNHY968eQgJCUFkZCSCg4NhZmaGsWPHAqj+ukJE7xkZEVEDs3PnThmAKl+WlpZy61haWsrGjRsnvPfx8ZF17Nixyu2sW7dOBkCWkpIitzw5OVkGQDZt2jS55efPn5cBkH3xxRcymUwme/bsmUxdXV02cuRIuXrx8fEyADIXFxdh2alTp2QAZL179652/1+9eiV7+fKlrG/fvrKhQ4cKy1NSUmQAZB06dJAVFxcLyzds2CADIBs0aJBcO3PmzJEBkGVlZclkMpns119/lQGQJSUlVRtDWYoeD5lMJgsJCZEBkD158qTadkvrvvlq2rSp7NChQ3J1S/d9586d5dZfu3atXN1p06bJmjRpIispKZHJZDJZVFRUhfUOHDggAyD77rvvhGWWlpYyDQ0N2YMHD4RlSUlJMgAyU1NTWW5urrD88OHDMgCyI0eOlIupMiUlJbKXL1/KYmJiZABkV65cUXhdmUwmKy4ulpmZmck6d+4s7J9MJpPdvXtX1rhx43KfCwCykJAQ4b29vb1syJAhVW5j+vTplcYBQKarqyt79uxZhWVlt1X6OS57DstkMtm5c+dkAGQrVqwQlr35+S3l4uIi9zlKTEwsdx6UevP4iTlvXVxcZABk58+fl6vbtm1bmaenp/BekesKEb0/OLSPiBqsH3/8EYmJieVeH3/8cbXrduvWDVeuXMG0adPwn//8B9nZ2Qpv99SpUwBQbqhRt27dYGdnh5MnTwIAEhISUFhYiBEjRsjV69GjR7nZ00oNGzaswuVbt25F586d0aRJE6iqqqJx48Y4efIkkpOTy9Xt37+/XM+DnZ0dAAi/0L+5/P79+wCAjh07Qk1NDQEBAdi9e3e54VWVUfR4KOv3339HYmIiLly4gKNHj8Ld3R2ffvopIiMjFVp/0KBBcu/bt2+PgoICYVhjaa/Pm/F/8skn0NLSKhd/x44d0bx5c+F96XF0dXWV67UrXX7v3r0q4/vrr7/g5+cHqVQKFRUVNG7cGC4uLgBQ4d+3Kjdv3sSjR4/g5+cnN4zN0tISzs7O1a7frVs3HD9+HAsXLsTp06eRn58vavsA0KdPH+jr6ytcf/To0XLvnZ2dYWlpKZxXtUXseSuVStGtWze5Ze3bt5f7+77NdYWIGh4mUkTUYNnZ2cHR0bHcq/QemqoEBwfj66+/RkJCAry9vWFoaIi+ffsqNBVz6TCnssOZSpmZmQnlpf81MTEpV6+iZZW1uX79evzf//0funfvjoMHDyIhIQGJiYnw8vKq8IuugYGB3Hs1NbUqlxcUFAAArK2t8fvvv8PY2BjTp0+HtbU1rK2tq72/Q9HjoawOHTrA0dERXbt2xYABA/DLL7/AxsYG06dPV2h9Q0NDufelw/RKj93Tp0+hqqqKZs2aydWTSCSQSqXl4lf2+FYkJycHvXr1wvnz57FixQqcPn0aiYmJOHTokFyMiiqNVSqVliuraNmbvvnmGyxYsACHDx+Gm5sbDAwMMGTIENy+fVvhGCo6D6pSWaxve95UR+x5++Z5BLw+l8r+jd7mukJEDQ8TKSL6IKmqqiIwMBCXL1/Gs2fPsH//fqSmpsLT07PaiRVKv1ClpaWVK3v06JFwX0VpvcePH5erl56eXmHbFT3n5qeffoKrqyu2bNmCAQMGoHv37nB0dMSLFy+q3kkl9OrVC//617+QlZWFhIQEODk5Yc6cOfj5558rXUfR41FTGjVqhHbt2iEtLa3cZBnKMDQ0xKtXr4R70UrJZDKkp6fXePxlRUdH49GjR9ixYwcmTZqE3r17w9HRETo6Okq1V/q3qOj8quycK0tLSwtLly7FH3/8gfT0dGzZsgUJCQkYOHCgwjGIfVZTZbGWTVyaNGmCwsLCcvX+/vtvUdsqqzbO27e5rhBRw8NEiog+eHp6ehg+fDimT5+OZ8+eCTOQvdlzUapPnz4AXic4ZSUmJiI5ORl9+/YFAHTv3h3q6uo4cOCAXL2EhIRqh3uVJZFIyj1v6erVq3I3+9c0FRUVdO/eXZih7fLly5XWVfR41JTi4mJcu3YN6urqop8NVpHS+N6M/+DBg8jNza3x+MsqTTre/Ptu27ZNqfZat24NU1NT7N+/X27SlXv37iEuLk5UWyYmJvD398eoUaNw8+ZNIRGo7HOhrL1798q9j4uLw7179+QmUbGyssLVq1fl6t26dQs3b96UWyYmtto+byu7rhDR+4Oz9hHRB2ngwIGwt7eHo6MjmjVrhnv37mHDhg2wtLSEra0tAMDBwQEAsHHjRowbNw6NGzdG69at0bp1awQEBGDTpk1o1KgRvL29hdm+zM3NMXfuXACvh3oFBgZi9erV0NfXx9ChQ/HgwQMsXboUpqam1c6gVsrHxwfLly9HSEgIXFxccPPmTSxbtgwtW7ascNZCZW3duhXR0dEYMGAALCwsUFBQIMxs5+7uXul6ih4PZV26dEkYrvn48WPs2LEDf/zxB+bOnYsmTZq8VdsA0K9fP3h6emLBggXIzs5Gz549hVn7OnXqhDFjxrz1Nirj7OwMfX19TJ06FSEhIWjcuDH27t2LK1euKNVeo0aNsHz5ckyaNAlDhw7F5MmT8fz5c4SGhio0tK979+7w8fFB+/btoa+vj+TkZOzZswdOTk7C/V+ln4uvvvoK3t7eUFFRQfv27YWhjGJdvHgRkyZNwieffILU1FQsWrQIzZs3x7Rp04Q6Y8aMwWeffYZp06Zh2LBhuHfvHtauXVtuOKa1tTU0NDSwd+9e2NnZQVtbG2ZmZjAzMyu33do4bxW5rhDR+4OJFBF9kNzc3HDw4EH88MMPyM7OhlQqRb9+/fDll18KzydydXVFcHAwdu/eje+//x4lJSU4deqUMMzO2toa27dvx7fffgtdXV14eXlh9erVckOSVq5cCS0tLWzduhU7d+5EmzZtsGXLFixatEjhqZ0XLVqEvLw8bN++HWvXrkXbtm2xdetWREZGCs+1qgkdO3bEb7/9hpCQEKSnp0NbWxv29vY4cuQIPDw8qlxX0eOhDC8vL+HfBgYGsLW1xY4dOzBu3Li3areURCLB4cOHERoaip07d2LlypUwMjLCmDFjsGrVqnK9RTXJ0NAQ//73vzFv3jx89tln0NLSwuDBg3HgwAHhYcNiTZw4EcDrRMfX1xdWVlb44osvEBMTU+350qdPHxw5cgTh4eHIy8tD8+bNMXbsWCxatEio4+fnh3PnziEiIgLLli2DTCZDSkpKpROoVGf79u3Ys2cPPv30UxQWFsLNzQ0bN26Uu+fMz88Pjx49Ej5H9vb22LJlC5YuXSrXlqamJnbs2IGlS5fCw8MDL1++REhISKXPK6vp81aR6woRvT8kMpkCD1whIqIak5KSgjZt2iAkJARffPFFXYdDRERESmAiRURUi65cuYL9+/fD2dkZTZs2xc2bN7F27VpkZ2fj+vXrlc7eR0RERPUbh/YREdUiLS0tXLx4Edu3b8fz58+hq6sLV1dXrFy5kkkUERFRA8YeKSIiIiIiIpE4/TkREREREZFITKSIiIiIiIhEYiJFREREREQkEiebAFBSUoJHjx5BR0dHeMo8ERERERF9eGQyGV68eAEzMzM0alR5vxMTKQCPHj2Cubl5XYdBRERERET1RGpqKlq0aFFpORMpADo6OgBeH6ymTZvWcTRERERERFRXsrOzYW5uLuQIlWEiBQjD+Zo2bcpEioiIiIiIqr3lh5NNEBERERERicREioiIiIiISCQmUkRERERERCLxHikiIqIGRCaT4dWrVyguLq7rUIiIGiQVFRWoqqq+9WOPmEgRERE1EEVFRUhLS0NeXl5dh0JE1KBpamrC1NQUampqSrfBRIqIiKgBKCkpQUpKClRUVGBmZgY1NTU+RJ6ISCSZTIaioiI8efIEKSkpsLW1rfKhu1VhIkVERNQAFBUVoaSkBObm5tDU1KzrcIiIGiwNDQ00btwY9+7dQ1FREZo0aaJUO5xsgoiIqAFR9pdTIiL6f2riWsqrMRERERERkUhMpIiIiIiIiERiIkVERER16u7du5BIJEhKSqrrUBAaGoqOHTuKWkcikeDw4cNV1nn69CmMjY1x9+5dpWNT1unTpyGRSPD8+XOF13F1dcWcOXNqLSZqeBrKOVFYWAgLCwtcunSp1rfFySaIlBR+4la1deb2a/UOIiGiD5ki16KaJPa65u/vj927dwvvDQwM0LVrV6xduxbt27cHAJibmyMtLQ1GRkY1GqsygoKCMHPmzBpvd/Xq1Rg4cCCsrKxqpL3Q0FAcPnxYoeTT2dkZaWlp0NXVrZFtv63vvvsO+/btw+XLl/HixQtkZmZCT09PKL979y6WL1+O6OhopKenw8zMDJ999hkWLVr0VlNVU/2TmZmJWbNm4ciRIwCAQYMGYdOmTXLnQ1WmTJmC7777DuHh4UKSp66ujqCgICxYsAC///57LUX+GnukiIiIqFZ5eXkhLS0NaWlpOHnyJFRVVeHj4yOUq6ioQCqVQlW17n/f1dbWhqGhYY22mZ+fj+3bt2PSpEk12q4iXr58CTU1NUil0nozXX5eXh68vLzwxRdfVFj+xx9/oKSkBNu2bcONGzcQHh6OrVu3Vlqf3k5RUVGdbdvPzw9JSUmIiopCVFQUkpKSMGbMGIXWPXz4MM6fPw8zM7NyZaNHj0ZsbCySk5NrOmQ5TKSIiIioVqmrq0MqlUIqlaJjx45YsGABUlNT8eTJEwDlh/aVDkU7efIkHB0doampCWdnZ9y8eVOu3S1btsDa2hpqampo3bo19uzZI1cukUiwbds2+Pj4QFNTE3Z2doiPj8edO3fg6uoKLS0tODk54c8//xTWeXNoX2JiIvr16wcjIyPo6urCxcUFly9fFrX/x48fh6qqKpycnIRliu5jRXbt2oWlS5fiypUrkEgkkEgk2LVrl7DPW7duxeDBg6GlpYUVK1aUG9r39OlTjBo1Ci1atICmpiYcHBywf//+KrcZEREBW1tbNGnSBCYmJhg+fLioY1DWnDlzsHDhQvTo0aPCci8vL+zcuRMeHh746KOPMGjQIAQFBeHQoUOitnPlyhW4ublBR0cHTZs2RZcuXXDx4kUAFQ/h3LBhQ7kewx07dqBdu3ZQV1eHqakpZsyYIZQ9f/4cAQEBMDExQZMmTWBvb4+jR48K5XFxcejduzc0NDRgbm6OWbNmITc3Vyiv6pj++uuvcHBwgIaGBgwNDeHu7i63bmX8/f0xZMgQLF26FMbGxmjatCmmTJkilyy5urpixowZCAwMhJGREfr16wcAiImJQbdu3YR9XbhwIV69eiXX/qtXrzBjxgzo6enB0NAQixcvhkwmqzauiiQnJyMqKgo//PADnJyc4OTkhO+//x5Hjx6t9nPw8OFDzJgxA3v37kXjxo3LlRsaGsLZ2bna8/ptMZEiIiKidyYnJwd79+6FjY1NtT0/ixYtQlhYGC5evAhVVVVMmDBBKIuMjMTs2bMxb948XL9+HVOmTMH48eNx6tQpuTaWL1+OsWPHIikpCW3atIGfnx+mTJmC4OBg4Ut12S/Hb3rx4gXGjRuH2NhYJCQkwNbWFv3798eLFy8U3uczZ87A0dFR9D5WZuTIkZg3bx7atWsn9PSNHDlSKA8JCcHgwYNx7dq1CtsrKChAly5dcPToUVy/fh0BAQEYM2YMzp8/X+H2Ll68iFmzZmHZsmW4efMmoqKi0Lt3b6F81apV0NbWrvIVGxtb7X5VJSsrCwYGBqLWGT16NFq0aIHExERcunQJCxcurPBLd2W2bNmC6dOnIyAgANeuXcORI0dgY2MD4PUDsr29vREXF4effvoJ//vf/7BmzRqoqKgAAK5duwZPT0/4+vri6tWrOHDgAM6ePSuca1Ud07S0NIwaNQoTJkxAcnIyTp8+DV9fX4UTlpMnTyI5ORmnTp3C/v37ERkZiaVLl8rV2b17N1RVVXHu3Dls27YNDx8+RP/+/dG1a1dcuXIFW7Zswfbt27FixYoK1zt//jy++eYbhIeH44cffhDKp06dWu25cP/+fQBAfHw8dHV10b17d2H9Hj16QFdXF3FxcZXuX0lJCcaMGYP58+ejXbt2ldbr1q3bW5931an7PnQiIiJ6rx09ehTa2toAgNzcXJiamuLo0aPVPsdl5cqVcHFxAQAsXLgQAwYMQEFBAZo0aYKvv/4a/v7+mDZtGgAgMDAQCQkJ+Prrr+Hm5ia0MX78eIwYMQIAsGDBAjg5OeHLL7+Ep6cnAGD27NkYP358pTH06dNH7v22bdugr6+PmJgYueGJVbl7926Fw4+q28fKaGhoQFtbG6qqqpBKpeXK/fz85BKolJQUufLmzZsjKChIeD9z5kxERUXhl19+kftSW+r+/fvQ0tKCj48PdHR0YGlpiU6dOgnlU6dOFY5xZZo3b15leVX+/PNPbNq0CWFhYaLWu3//PubPn482bdoAAGxtbUWtv2LFCsybNw+zZ88WlnXt2hUA8Pvvv+PChQtITk5Gq1av7xv86KOPhHrr1q2Dn5+fcN+Ora0tvvnmG7i4uGDLli1VHtO0tDS8evUKvr6+sLS0BAA4ODgoHLeamhp27NgBTU1NtGvXDsuWLcP8+fOxfPly4TNnY2ODtWvXCussWrQI5ubm2Lx5MyQSCdq0aYNHjx5hwYIFWLJkibCeubk5wsPDIZFI0Lp1a1y7dg3h4eGYPHkyAGDZsmVy51ZFSj8L6enpMDY2LldubGyM9PT0Stf/6quvoKqqilmzZlW5nebNm9f65C5MpIiIiKhWubm5YcuWLQCAZ8+eISIiAt7e3rhw4YLwRbEipZNRAICpqSkAICMjAxYWFkhOTkZAQIBc/Z49e2Ljxo2VtmFiYgJA/kupiYkJCgoKkJ2djaZNm5aLISMjA0uWLEF0dDQeP36M4uJi5OXlCb+qKyI/P7/SxKiqfVRWZb1fpYqLi7FmzRocOHAADx8+RGFhIQoLC6GlpVVh/X79+sHS0hIfffQRvLy84OXlhaFDh0JTUxPA6wlExPYWKerRo0fw8vLCJ598Ivoes8DAQEyaNAl79uyBu7s7PvnkE1hbWyu0bkZGBh49eoS+fftWWJ6UlIQWLVoISdSbLl26hDt37mDv3r3CMplMhpKSEqSkpFR5TDt06IC+ffvCwcEBnp6e8PDwwPDhw6Gvr69Q7B06dBD+NgDg5OSEnJwcpKamCp+3N8+R5ORkODk5yd1H17NnT+Tk5ODBgwfC+dijRw+5Ok5OTggLC0NxcTFUVFRgbGxcYXJUmYru25PJZJXez3fp0iVs3LgRly9frvaePw0NDeTl5SkcizI4tI+IiIhqlZaWFmxsbGBjY4Nu3bph+/btyM3Nxffff1/lemWHYZV+aSopKSm3rFRFX8AqaqO6dsvy9/fHpUuXsGHDBsTFxSEpKQmGhoaibtA3MjJCZmZmhWViYlFUZQlRqbCwMISHh+Pzzz9HdHQ0kpKS4OnpWek+6ejo4PLly9i/fz9MTU2xZMkSdOjQQbjnqraG9j169Ahubm5wcnLCd999J3r90NBQ3LhxAwMGDEB0dDTatm2LyMhIAECjRo3KDZV7+fKl8G8NDY0q266uvKSkBFOmTEFSUpLwunLlCm7fvg1ra+sqj6mKigpOnDiB48ePo23btti0aRNat25drmdRrLKfjTfPkYo+O6XHR8wkJWKG9kmlUjx+/LhcG0+ePBF+9HhTbGys8EODqqoqVFVVce/ePcybN6/c/W3Pnj1Ds2bNFI5dGeyRIiIiondKIpGgUaNGyM/PV7oNOzs7nD17FmPHjhWWxcXFwc7OriZCFMTGxiIiIgL9+/cHAKSmpuLvv/8W1UanTp3w008/1WhcampqKC4uVmrd2NhYDB48GJ999hmA11/6b9++XeWxU1VVhbu7O9zd3RESEgI9PT1ER0fD19e3Vob2PXz4EG5ubujSpQt27txZ7TDQyrRq1QqtWrXC3LlzMWrUKOzcuRNDhw5Fs2bNkJ6eLpdAlJ1KXkdHB1ZWVjh58qTcUNFS7du3x4MHD3Dr1q0Ke6U6d+6MGzduCPdUVaSqYyqRSNCzZ0/07NkTS5YsgaWlJSIjIxEYGFjtPl+5cgX5+flCspeQkABtbW20aNGi0nXatm2LgwcPyh2PuLg46OjoyP3tEhIS5NYrvW+w9N4wMUP7nJyckJWVhQsXLqBbt24AgPPnzyMrKwvOzs4VrjtmzBi4u7vLLfP09MSYMWPKDdG9fv263BDU2sBEioiIiGpVYWGhcM9DZmYmNm/ejJycHAwcOFDpNufPn48RI0agc+fO6Nu3L/71r3/h0KFDNf7cGBsbG+zZsweOjo7Izs7G/Pnzq+2NeJOnpyeCg4ORmZmp8PCs6lhZWSElJUUYYqajowN1dXWF1rWxscHBgwcRFxcHfX19rF+/Hunp6ZUmUkePHsVff/2F3r17Q19fH8eOHUNJSQlat24NQPzQvvT0dKSnp+POnTsAXk/MoKOjAwsLCxgYGODRo0dwdXWFhYUFvv76a2F2RwAV3hNWkfz8fMyfPx/Dhw9Hy5Yt8eDBAyQmJmLYsGEAXs9c9+TJE6xduxbDhw9HVFQUjh8/Lje8MzQ0FFOnToWxsTG8vb3x4sULnDt3DjNnzoSLiwt69+6NYcOGYf369bCxscEff/wBiUQCLy8vLFiwAD169MD06dMxefJkaGlpITk5GSdOnMCmTZuqPKbnz5/HyZMn4eHhAWNjY5w/fx5PnjxR+EeCoqIiTJw4EYsXL8a9e/cQEhKCGTNmVJmMTps2DRs2bMDMmTMxY8YM3Lx5EyEhIQgMDJRbLzU1FYGBgZgyZQouX75c7t41MUP77Ozs4OXlhcmTJ2Pbtm0AgICAAPj4+AjnFgC0adMGq1evxtChQ2FoaFhukprGjRtDKpXKrQO8/sFg+fLlCsWiLCZSREREDVhDePB3VFSUcP+Pjo4O2rRpg19++QWurq5KtzlkyBBs3LgR69atw6xZs9CyZUvs3LnzrdqsyI4dOxAQEIBOnTrBwsICq1atqvYX9zc5ODjA0dER//jHPzBlypQaiWvYsGE4dOgQ3Nzc8Pz5c+zcuRP+/v4Krfvll18iJSUFnp6e0NTUREBAAIYMGYKsrKwK6+vp6eHQoUMIDQ1FQUEBbG1tsX///ipnTKvK1q1b5WaRK52trnQffvvtN9y5cwd37twp14tSdjielZUV/P39ERoaWm4bKioqePr0KcaOHYvHjx/DyMgIvr6+wnbt7OwQERGBVatWYfny5Rg2bBiCgoLkhhCOGzcOBQUFCA8PR1BQEIyMjOSmKD948CCCgoIwatQo5ObmwsbGBmvWrAHwuscqJiYGixYtQq9evSCTyWBtbS3MrljVMU1OTsaZM2ewYcMGZGdnw9LSEmFhYfD29lbo+Pbt2xe2trbo3bs3CgsL8emnn1Z4jMpq3rw5jh07hvnz56NDhw4wMDAQkrGyxo4di/z8fHTr1g0qKiqYOXNmuXsVxdi7dy9mzZoFDw8PAK8fyLt582a5Ojdv3qz03KxMfHw8srKy3mqafkVIZMpO/v4eyc7Ohq6uLrKysiq80ZSoIuEnblVbpyF8wSGihqGgoAApKSlo2bJllTO6Uf107NgxBAUF4fr160oPU6P/Jz8/HwYGBjh27FiFQ+8+VP7+/nj+/DkOHz5c16HUqU8++QSdOnWq8iHOVV1TFc0N2CNFREREVMv69++P27dv4+HDhzA3N6/rcBq8mJgY9OnTh0kUlVNYWIgOHTpg7ty5tb4tJlJERERE70DZ5xFVp127drh3716FZdu2bcPo0aNrKqwGqXTK8A9N6fPYKnL8+PF3GEn9pa6uXm5IYm1hIkVERERUzxw7dkxuOu6yKpsamt5/ZWcWfFPz5s3Rq1evdxcMMZEiIiIiqm+qelAxfbiqmk6d3j3e7UhERERERCQSEykiIiIiIiKRmEgRERERERGJxESKiIiIiIhIJCZSREREREREIjGRIiIiojp19+5dSCSSKqd2fldCQ0PRsWNHUetIJBIcPny4yjpPnz6FsbEx7t69q3Rsyjp9+jQkEgmeP3+u8Dqurq6YM2dOrcVEDU9DOScKCwthYWGBS5cu1fq2OP05ERFRQ3Zq9bvdnluwqOr+/v7YvXu38N7AwABdu3bF2rVr0b59ewCAubk50tLSYGRkVKOhKiMoKAgzZ86s8XZXr16NgQMHwsrKqkbaCw0NxeHDhxVKPp2dnZGWlgZdXd0a2fbb+u6777Bv3z5cvnwZL168QGZmJvT09OTqWFlZlXsg8YIFC7BmzZp3GCnVtszMTMyaNQtHjhwBAAwaNAibNm0qdz6U9eY1BQC6d++OhIQEAK8fyBsUFIQFCxbg999/r7XYAfZIERERUS3z8vJCWloa0tLScPLkSaiqqsLHx0coV1FRgVQqhapq3f++q62tDUNDwxptMz8/H9u3b8ekSZNqtF1FvHz5EmpqapBKpZBIJO98+xXJy8uDl5cXvvjiiyrrLVu2TDhv0tLSsHjx4ncU4YelqKiozrbt5+eHpKQkREVFISoqCklJSRgzZky165W9pqSlpeHYsWNy5aNHj0ZsbCySk5NrK3QATKSIiIiolqmrq0MqlUIqlaJjx45YsGABUlNT8eTJEwDlh/aVDkU7efIkHB0doampCWdnZ9y8eVOu3S1btsDa2hpqampo3bo19uzZI1cukUiwbds2+Pj4QFNTE3Z2doiPj8edO3fg6uoKLS0tODk54c8//xTWeXNoX2JiIvr16wcjIyPo6urCxcUFly9fFrX/x48fh6qqKpycnIRliu5jRXbt2oWlS5fiypUrkEgkkEgk2LVrl7DPW7duxeDBg6GlpYUVK1aUG9r39OlTjBo1Ci1atICmpiYcHBywf//+KrcZEREBW1tbNGnSBCYmJhg+fLioY1DWnDlzsHDhQvTo0aPKejo6OsJ5I5VKoa2tLWo7V65cgZubG3R0dNC0aVN06dIFFy9eBFDxEM4NGzaU6zHcsWMH2rVrB3V1dZiammLGjBlC2fPnzxEQEAATExM0adIE9vb2OHr0qFAeFxeH3r17Q0NDA+bm5pg1axZyc3OF8qqO6a+//goHBwdoaGjA0NAQ7u7ucutWxt/fH0OGDMHSpUthbGyMpk2bYsqUKXLJkqurK2bMmIHAwEAYGRmhX79+AICYmBh069ZN2NeFCxfi1atXcu2/evUKM2bMgJ6eHgwNDbF48WLIZLJq46pIcnIyoqKi8MMPP8DJyQlOTk74/vvvcfTo0Wo/B2WvKVKpFAYGBnLlhoaGcHZ2rva8fltMpIiIiOidycnJwd69e2FjY1Ntz8+iRYsQFhaGixcvQlVVFRMmTBDKIiMjMXv2bMybNw/Xr1/HlClTMH78eJw6dUqujeXLl2Ps2LFISkpCmzZt4OfnhylTpiA4OFj4Ul32y/GbXrx4gXHjxiE2NhYJCQmwtbVF//798eLFC4X3+cyZM3B0dBS9j5UZOXIk5s2bh3bt2gm/yI8cOVIoDwkJweDBg3Ht2rUK2ysoKECXLl1w9OhRXL9+HQEBARgzZgzOnz9f4fYuXryIWbNmYdmyZbh58yaioqLQu3dvoXzVqlXQ1tau8hUbG1vtfr3pq6++gqGhITp27IiVK1eK7jkZPXo0WrRogcTERFy6dAkLFy5E48aNFV5/y5YtmD59OgICAnDt2jUcOXIENjY2AICSkhJ4e3sjLi4OP/30E/73v/9hzZo1UFFRAQBcu3YNnp6e8PX1xdWrV3HgwAGcPXtWONeqOqZpaWkYNWoUJkyYgOTkZJw+fRq+vr4KJywnT55EcnIyTp06hf379yMyMhJLly6Vq7N7926oqqri3Llz2LZtGx4+fIj+/fuja9euuHLlCrZs2YLt27djxYoVFa53/vx5fPPNNwgPD8cPP/wglE+dOrXac+H+/fsAgPj4eOjq6qJ79+7C+j169ICuri7i4uKq3MfTp0/D2NgYrVq1wuTJk5GRkVGuTrdu3ZQ678So+z50IiIieq8dPXpU6E3Izc2Fqakpjh49ikaNqv49d+XKlXBxcQEALFy4EAMGDEBBQQGaNGmCr7/+Gv7+/pg2bRoAIDAwEAkJCfj666/h5uYmtDF+/HiMGDECwOt7bJycnPDll1/C09MTADB79myMHz++0hj69Okj937btm3Q19dHTEyM3PDEqty9exdmZmai97EyGhoa0NbWhqqqKqRSablyPz8/uQQqJSVFrrx58+YICgoS3s+cORNRUVH45Zdf5L7Ulrp//z60tLTg4+MDHR0dWFpaolOnTkL51KlThWNcmebNm1dZ/qbZs2ejc+fO0NfXx4ULFxAcHIyUlBS5L+3VuX//PubPn482bdoAAGxtbUXFsGLFCsybNw+zZ88WlnXt2hUA8Pvvv+PChQtITk5Gq1atAAAfffSRUG/dunXw8/MTJmewtbXFN998AxcXF2zZsqXKY5qWloZXr17B19cXlpaWAAAHBweF41ZTU8OOHTugqamJdu3aYdmyZZg/fz6WL18ufOZsbGywdu1aYZ1FixbB3NwcmzdvhkQiQZs2bfDo0SMsWLAAS5YsEdYzNzdHeHg4JBIJWrdujWvXriE8PByTJ08G8Ho4ZtlzqyKln4X09HQYGxuXKzc2NkZ6enql63t7e+OTTz6BpaUlUlJS8OWXX6JPnz64dOkS1NXVhXrNmzev9cldmEgRUf2i6I3zIm94J6K64+bmhi1btgAAnj17hoiICHh7e+PChQvCF8WKlE5GAQCmpqYAgIyMDFhYWCA5ORkBAQFy9Xv27ImNGzdW2oaJiQkA+S+lJiYmKCgoQHZ2Npo2bVouhoyMDCxZsgTR0dF4/PgxiouLkZeXJ/yqroj8/PxKE6Oq9lFZlfV+lSouLsaaNWtw4MABPHz4EIWFhSgsLISWllaF9fv16wdLS0t89NFH8PLygpeXF4YOHQpNTU0ArycQeXNo1duaO3eu8O/27dtDX18fw4cPF3qpFBEYGIhJkyZhz549cHd3xyeffAJra2uF1s3IyMCjR4/Qt2/fCsuTkpLQokULIYl606VLl3Dnzh3s3btXWCaTyVBSUoKUlJQqj2mHDh3Qt29fODg4wNPTEx4eHhg+fDj09fUVir1Dhw7C3wYAnJyckJOTg9TUVOHz9uY5kpycDCcnJ7n76Hr27ImcnBw8ePBAOB979OghV8fJyQlhYWEoLi6GiooKjI2NK0yOKlPRfXsymazK+/nK9r7a29vD0dERlpaW+Pe//w1fX1+hTENDA3l5eQrHogwO7SMiIqJapaWlBRsbG9jY2KBbt27Yvn07cnNz8f3331e5XtlhWKVfrEpKSsotK1XRF7CK2qiu3bL8/f1x6dIlbNiwAXFxcUhKSoKhoaGoYWZGRkbIzMyssExMLIqqLCEqFRYWhvDwcHz++eeIjo5GUlISPD09K90nHR0dXL58Gfv374epqSmWLFmCDh06CPdc1dbQvrJK76e6c+eOwuuEhobixo0bGDBgAKKjo9G2bVtERkYCABo1alRuqNzLly+Ff2toaFTZdnXlJSUlmDJlCpKSkoTXlStXcPv2bVhbW1d5TFVUVHDixAkcP34cbdu2xaZNm9C6detyPYtilf1svHmOVPTZKT0+YiYpETO0TyqV4vHjx+XaePLkifCjhyJMTU1haWmJ27dvyy1/9uwZmjVrpnA7ymCPFBEREb1TEokEjRo1Qn5+vtJt2NnZ4ezZsxg7dqywLC4uDnZ2djURoiA2NhYRERHo378/ACA1NRV///23qDY6deqEn376qUbjUlNTQ3FxsVLrxsbGYvDgwfjss88AvP7Sf/v27SqPnaqqKtzd3eHu7o6QkBDo6ekhOjoavr6+tTK0703//e9/Afy/XjtFtWrVCq1atcLcuXMxatQo7Ny5E0OHDkWzZs2Qnp4ul0CUnUpeR0cHVlZWOHnypNxQ0VLt27fHgwcPcOvWrQp7pTp37owbN24I91RVpKpjKpFI0LNnT/Ts2RNLliyBpaUlIiMjERgYWO0+X7lyBfn5+UKyl5CQAG1tbbRo0aLSddq2bYuDBw/KHY+4uDjo6OjI/e1Kpxgv+97W1la4N0zM0D4nJydkZWXhwoUL6NatGwDg/PnzyMrKgrOzc7X7Werp06dITU0td25cv35dbghqbWAiRURERLWqsLBQuOchMzMTmzdvRk5ODgYOHKh0m/Pnz8eIESPQuXNn9O3bF//6179w6NChGn9ujI2NDfbs2QNHR0dkZ2dj/vz51fZGvMnT0xPBwcHIzMxUeHhWdaysrJCSkiIMMdPR0ZG7P6QqNjY2OHjwIOLi4qCvr4/169cjPT290kTq6NGj+Ouvv9C7d2/o6+vj2LFjKCkpQevWrQGIH9qXnp6O9PR0oXfp2rVr0NHRgYWFBQwMDBAfH4+EhAS4ublBV1cXiYmJmDt3LgYNGqTwkMf8/HzMnz8fw4cPR8uWLfHgwQMkJiZi2LBhAF7PXPfkyROsXbsWw4cPR1RUFI4fPy43vDM0NBRTp06FsbExvL298eLFC5w7dw4zZ86Ei4sLevfujWHDhmH9+vWwsbHBH3/8AYlEAi8vLyxYsAA9evTA9OnTMXnyZGhpaSE5ORknTpzApk2bqjym58+fx8mTJ+Hh4QFjY2OcP38eT548UfhHgqKiIkycOBGLFy/GvXv3EBISghkzZlR5T+K0adOwYcMGzJw5EzNmzMDNmzcREhKCwMBAufVSU1MRGBiIKVOm4PLly9i0aRPCwsKEcjFD++zs7ODl5YXJkydj27ZtAICAgAD4+PgI5xYAtGnTBqtXr8bQoUORk5OD0NBQDBs2DKamprh79y6++OILGBkZYejQoXLtx8bGYvny5QrFoiwmUkRERA1ZA7hfMCoqSvi1WEdHB23atMEvv/wCV1dXpdscMmQINm7ciHXr1mHWrFlo2bIldu7c+VZtVmTHjh0ICAhAp06dYGFhgVWrVlX7i/ubHBwc4OjoiH/84x+YMmVKjcQ1bNgwHDp0CG5ubnj+/Dl27twJf39/hdb98ssvkZKSAk9PT2hqaiIgIABDhgxBVlZWhfX19PRw6NAhhIaGoqCgALa2tti/fz/atWunVOxbt26Vm0WudLa60n1QV1fHgQMHsHTpUhQWFsLS0hKTJ0/G559/LteOlZUV/P39ERoaWm4bKioqePr0KcaOHYvHjx/DyMgIvr6+wnbt7OwQERGBVatWYfny5Rg2bBiCgoLw3XffCW2MGzcOBQUFCA8PR1BQEIyMjOSmKD948CCCgoIwatQo5ObmwsbGRnhgcPv27RETE4NFixahV69ekMlksLa2Fu7vqeqYJicn48yZM9iwYQOys7NhaWmJsLAweHt7K3R8+/btC1tbW/Tu3RuFhYX49NNPKzxGZTVv3hzHjh3D/Pnz0aFDBxgYGAjJWFljx45Ffn4+unXrBhUVFcycObPcvYpi7N27F7NmzYKHhweA1w/k3bx5s1ydmzdvCuemiooKrl27hh9//BHPnz+Hqakp3NzccODAAejo6AjrxMfHIysr662m6VeERKbs5O/vkezsbOjq6iIrK6vCG02JKhJ+4la1deb2q/gmVKoCJ5sgqlBBQQFSUlLQsmXLKmd0o/rp2LFjCAoKwvXr16udrZCql5+fDwMDAxw7dqzCoXcfKn9/fzx//hyHDx+u61Dq1CeffIJOnTpV+dDnqq6piuYG7JEiIiIiqmX9+/fH7du38fDhQ5ibm9d1OA1eTEwM+vTpwySKyiksLESHDh3kZn6sLUykiIiIiN6Bss8jqk67du1w7969Csu2bduG0aNH11RYDVLplOEfmtLnsVXk+PHj7zCS+ktdXb3ckMTawkSKiIiIqJ45duyY3HTcZYmZGpreL2VnFnxT8+bN0atXr3cXDDGRIiIiIqpvqnpQMX24qppOnd493u1IREREREQkEhMpIiIiIiIikZhIERERERERicREioiIiIiISCQmUkRERERERCIxkSIiIqI6dffuXUgkkiqndn5XQkND0bFjR1HrSCQSHD58uMo6T58+hbGxMe7evat0bMo6ffo0JBIJnj9/rvA6rq6umDNnTq3FRA1PQzknCgsLYWFhgUuXLtX6tjj9OdF7IPzErWrrzO3X6h1EQkTvWkRSxDvd3rSO00TV9/f3x+7du4X3BgYG6Nq1K9auXYv27dsDAMzNzZGWlgYjI6MajVUZQUFBmDlzZo23u3r1agwcOBBWVlY10l5oaCgOHz6sUPLp7OyMtLQ06Orq1si238azZ88QEhKC3377DampqTAyMsKQIUOwfPlyufisrKzKPZB4wYIFWLNmzbsOmWpRZmYmZs2ahSNHjgAABg0ahE2bNkFPT6/SdXJycrBw4UIcPnwYT58+hZWVFWbNmoX/+7//A/D6gbxBQUFYsGABfv/991qNv057pFavXo2uXbtCR0cHxsbGGDJkCG7evClXRyaTITQ0FGZmZtDQ0ICrqytu3LghV6ewsBAzZ86EkZERtLS0MGjQIDx48OBd7goRERFVwsvLC2lpaUhLS8PJkyehqqoKHx8foVxFRQVSqRSqqnX/+662tjYMDQ1rtM38/Hxs374dkyZNqtF2FfHy5UuoqalBKpVCIpG88+2/6dGjR3j06BG+/vprXLt2Dbt27UJUVBQmTpxYru6yZcuE8yYtLQ2LFy+ug4jff0VFRXW2bT8/PyQlJSEqKgpRUVFISkrCmDFjqlxn7ty5iIqKwk8//YTk5GTMnTsXM2fOxD//+U+hzujRoxEbG4vk5ORajb9OE6mYmBhMnz4dCQkJOHHiBF69egUPDw/k5uYKddauXYv169dj8+bNSExMhFQqRb9+/fDixQuhzpw5cxAZGYmff/4ZZ8+eRU5ODnx8fFBcXFwXu0VERERlqKurQyqVQiqVomPHjliwYAFSU1Px5MkTAOWH9pUORTt58iQcHR2hqakJZ2fncj+2btmyBdbW1lBTU0Pr1q2xZ88euXKJRIJt27bBx8cHmpqasLOzQ3x8PO7cuQNXV1doaWnByckJf/75p7DOm0P7EhMT0a9fPxgZGUFXVxcuLi64fPmyqP0/fvw4VFVV4eTkJCxTdB8rsmvXLixduhRXrlyBRCKBRCLBrl27hH3eunUrBg8eDC0tLaxYsaLc0L6nT59i1KhRaNGiBTQ1NeHg4ID9+/dXuc2IiAjY2tqiSZMmMDExwfDhw0Udg1L29vY4ePAgBg4cCGtra/Tp0wcrV67Ev/71L7x69Uquro6OjnDeSKVSaGtri9rWlStX4ObmBh0dHTRt2hRdunTBxYsXAVQ8hHPDhg3legx37NiBdu3aQV1dHaamppgxY4ZQ9vz5cwQEBMDExARNmjSBvb09jh49KpTHxcWhd+/e0NDQgLm5OWbNmiX3HbeqY/rrr7/CwcEBGhoaMDQ0hLu7u9y6lfH398eQIUOwdOlSGBsbo2nTppgyZYpcsuTq6ooZM2YgMDAQRkZG6NevH4DX38u7desm7OvChQvL/U1evXqFGTNmQE9PD4aGhli8eDFkMlm1cVUkOTkZUVFR+OGHH+Dk5AQnJyd8//33OHr0aJWfg/j4eIwbNw6urq6wsrJCQEAAOnToIPxtAcDQ0BDOzs7Vntdvq04TqaioKPj7+6Ndu3bo0KEDdu7cifv37wtjGmUyGTZs2IBFixbB19cX9vb22L17N/Ly8rBv3z4AQFZWFrZv346wsDC4u7ujU6dO+Omnn3Dt2rVa784jIiIicXJycrB3717Y2NhU2/OzaNEihIWF4eLFi1BVVcWECROEssjISMyePRvz5s3D9evXMWXKFIwfPx6nTp2Sa2P58uUYO3YskpKS0KZNG/j5+WHKlCkIDg4WvniV/XL8phcvXmDcuHGIjY1FQkICbG1t0b9/f7kfdKtz5swZODo6it7HyowcORLz5s1Du3bthN6akSNHCuUhISEYPHgwrl27VmF7BQUF6NKlC44ePYrr168jICAAY8aMwfnz5yvc3sWLFzFr1iwsW7YMN2/eRFRUFHr37i2Ur1q1Ctra2lW+YmNjK92frKwsNG3atFyP5FdffQVDQ0N07NgRK1euFN1zMnr0aLRo0QKJiYm4dOkSFi5ciMaNGyu8/pYtWzB9+nQEBATg2rVrOHLkCGxsbAAAJSUl8Pb2RlxcHH766Sf873//w5o1a6CiogIAuHbtGjw9PeHr64urV6/iwIEDOHv2rHCuVXVM09LSMGrUKEyYMAHJyck4ffo0fH19FU5YTp48ieTkZJw6dQr79+9HZGQkli5dKldn9+7dUFVVxblz57Bt2zY8fPgQ/fv3R9euXXHlyhVs2bIF27dvx4oVKypc7/z58/jmm28QHh6OH374QSifOnVqtefC/fv3AbxOiHR1ddG9e3dh/R49ekBXVxdxcXGV7t/HH3+MI0eO4OHDh5DJZDh16hRu3boFT09PuXrdunWr8ryrCXXfh15GVlYWgNfjpwEgJSUF6enp8PDwEOqoq6vDxcUFcXFxmDJlCi5duoSXL1/K1TEzM4O9vT3i4uLKHVTg9VDAwsJC4X12dnZt7RIREdEH7+jRo0JvQm5uLkxNTXH06FE0alT177krV66Ei4sLAGDhwoUYMGAACgoK0KRJE3z99dfw9/fHtGmv79kKDAxEQkICvv76a7i5uQltjB8/HiNGjADw+h4bJycnfPnll8L3g9mzZ2P8+PGVxtCnTx+599u2bYO+vj5iYmLkhidW5e7duzAzMxO9j5XR0NCAtrY2VFVVIZVKy5X7+fnJJVApKSly5c2bN0dQUJDwfubMmYiKisIvv/wi96W21P3796GlpQUfHx/o6OjA0tISnTp1EsqnTp0qHOPKNG/evMLlT58+xfLlyzFlyhS55bNnz0bnzp2hr6+PCxcuIDg4GCkpKXJf2qtz//59zJ8/H23atAEA2NraKrwuAKxYsQLz5s3D7NmzhWVdu3YFAPz++++4cOECkpOT0arV63uQP/roI6HeunXr4OfnJ0zOYGtri2+++QYuLi7YsmVLlcc0LS0Nr169gq+vLywtLQEADg4OCsetpqaGHTt2QFNTE+3atcOyZcswf/58LF++XPjM2djYYO3atcI6ixYtgrm5OTZv3gyJRII2bdrg0aNHWLBgAZYsWSKsZ25ujvDwcEgkErRu3RrXrl1DeHg4Jk+eDOD1cMyy51ZFSj8L6enpMDY2LldubGyM9PT0Stf/5ptvMHnyZLRo0QKqqqpo1KgRfvjhB3z88cdy9Zo3b17rk7vUm0RKJpMhMDAQH3/8Mezt7QFAOIgmJiZydU1MTIQbENPT06GmpgZ9ff1ydSr7I6xevbpcZk5ERES1w83NDVu2bAHwerKBiIgIeHt748KFC8IXxYqUTkYBAKampgCAjIwMWFhYIDk5GQEBAXL1e/bsiY0bN1baRun3ibJfSk1MTFBQUIDs7Gw0bdq0XAwZGRlYsmQJoqOj8fjxYxQXFyMvL0/4VV0R+fn5lSZGVe2jsirr/SpVXFyMNWvW4MCBA3j48KHwA7OWllaF9fv16wdLS0t89NFH8PLygpeXF4YOHQpNTU0Ar38AL/0RXIzs7GwMGDAAbdu2RUhIiFzZ3LlzhX+3b98e+vr6GD58uNBLpYjAwEBMmjQJe/bsgbu7Oz755BNYW1srtG5GRgYePXqEvn37VlielJSEFi1aCEnUmy5duoQ7d+5g7969wjKZTIaSkhKkpKRUeUw7dOiAvn37wsHBAZ6envDw8MDw4cPLfdetTIcOHYS/DQA4OTkhJycHqampwuftzXMkOTkZTk5OcvfR9ezZEzk5OXjw4IFwPvbo0UOujpOTE8LCwlBcXAwVFRUYGxtXmBxVpqL79mQyWZX3833zzTdISEjAkSNHYGlpiTNnzmDatGkwNTWFu7u7UE9DQwN5eXkKx6KMejP9+YwZM3D16tUKxzK+eTCrO8DV1QkODkZWVpbwSk1NVT5wIiIiqpKWlhZsbGxgY2ODbt26Yfv27cjNzcX3339f5Xplh2GV/j+9pKSk3LJSFf2/v6I2qmu3LH9/f1y6dAkbNmxAXFwckpKSYGhoKGqYmZGRETIzMyssExOLoipLiEqFhYUhPDwcn3/+OaKjo5GUlARPT89K90lHRweXL1/G/v37YWpqiiVLlqBDhw7CPVfKDO178eIFvLy8oK2tjcjIyGqH3PXo0QMAcOfOHQWPwuv7oG7cuIEBAwYgOjoabdu2RWRkJACgUaNG5YbKvXz5Uvi3hoZGlW1XV15SUoIpU6YgKSlJeF25cgW3b9+GtbV1lcdURUUFJ06cwPHjx9G2bVts2rQJrVu3LtezKFbZz8ab50hFn53S4yNmkhIxQ/ukUikeP35cro0nT56U60QplZ+fjy+++ALr16/HwIED0b59e8yYMQMjR47E119/LVf32bNnaNasmcKxK6Ne9EjNnDkTR44cwZkzZ9CiRQtheWl3dXp6uvArDfD6V4LSAyyVSlFUVITMzEy5TD0jIwPOzs4Vbk9dXR3q6uq1sStERERUDYlEgkaNGiE/P1/pNuzs7HD27FmMHTtWWBYXFwc7O7uaCFEQGxuLiIgI9O/fHwCQmpqKv//+W1Qbpfdv1yQ1NTWlJ9WKjY3F4MGD8dlnnwF4/aX/9u3bVR47VVVVuLu7w93dHSEhIdDT00N0dDR8fX1FD+3Lzs6Gp6cn1NXVceTIkSqHMZb673//CwBy3wcV0apVK7Rq1Qpz587FqFGjsHPnTgwdOhTNmjVDenq6XAJRdip5HR0dWFlZ4eTJk3JDRUu1b98eDx48wK1btyrslercuTNu3Lgh3FNVkaqOqUQiQc+ePdGzZ08sWbIElpaWiIyMRGBgYLX7fOXKFeTn5wvJXkJCArS1teW+Y7+pbdu2OHjwoNzxiIuLg46OjtzfLiEhQW690vsGS+8NEzO0z8nJCVlZWbhw4QK6desGADh//jyysrIq/Q7/8uVLvHz5stywYBUVlXI/QFy/fl1uCGptqNNESiaTYebMmYiMjMTp06fRsmVLufKWLVtCKpXixIkTwoEoKipCTEwMvvrqKwBAly5d0LhxY5w4cUL4EKelpeH69etyYz+JiIiobhQWFgrD7TMzM7F582bk5ORg4MCBSrc5f/58jBgxAp07d0bfvn3xr3/9C4cOHarxiaZsbGywZ88eODo6Ijs7G/Pnz6+2N+JNnp6eCA4OLvej79uwsrJCSkqKMMRMR0dH4R+JbWxscPDgQcTFxUFfXx/r169Henp6pYnU0aNH8ddff6F3797Q19fHsWPHUFJSgtatWwMQN7TvxYsX8PDwQF5eHn766SdkZ2cL96o3a9YMKioqiI+PR0JCAtzc3KCrq4vExETMnTsXgwYNUnjIY35+PubPn4/hw4ejZcuWePDgARITEzFs2DAAr2eue/LkCdauXYvhw4cjKioKx48flxveGRoaiqlTp8LY2Bje3t548eIFzp07h5kzZ8LFxQW9e/fGsGHDsH79etjY2OCPP/6ARCKBl5cXFixYgB49emD69OmYPHkytLS0kJycjBMnTmDTpk1VHtPz58/j5MmT8PDwgLGxMc6fP48nT54o/CNBUVERJk6ciMWLF+PevXsICQnBjBkzqrwncdq0adiwYQNmzpyJGTNm4ObNmwgJCUFgYKDceqmpqQgMDMSUKVNw+fJlbNq0CWFhYUK5mKF9dnZ28PLywuTJk7Ft2zYAQEBAAHx8fIRzCwDatGmD1atXY+jQoWjatClcXFyEz6GlpSViYmLw448/Yv369XLtx8bGYvny5QrFoqw6TaSmT5+Offv24Z///Cd0dHSEi6yuri40NDQgkUgwZ84crFq1Cra2trC1tcWqVaugqakJPz8/oe7EiRMxb948GBoawsDAAEFBQXBwcJAbJ0lERPQ+EvuA3LoQFRUl9CTo6OigTZs2+OWXX+Dq6qp0m0OGDMHGjRuxbt06zJo1Cy1btsTOnTvfqs2K7NixAwEBAejUqRMsLCywatWqan9xf5ODgwMcHR3xj3/8o9ykCsoaNmwYDh06BDc3Nzx//hw7d+6Ev7+/Qut++eWXSElJgaenJzQ1NREQEIAhQ4YIk369SU9PD4cOHUJoaCgKCgpga2uL/fv3o127dqLjvnTpkjA74Ju9NSkpKbCysoK6ujoOHDiApUuXorCwEJaWlpg8eTI+//xzufpWVlbw9/dHaGhoue2oqKjg6dOnGDt2LB4/fgwjIyP4+voK98jb2dkhIiICq1atwvLlyzFs2DAEBQXhu+++E9oYN24cCgoKEB4ejqCgIBgZGclNUX7w4EEEBQVh1KhRyM3NhY2NjfDA4Pbt2yMmJgaLFi1Cr169IJPJYG1tLcyuWNUxTU5OxpkzZ7BhwwZkZ2fD0tISYWFh8Pb2VugY9+3bF7a2tujduzcKCwvx6aefVniMymrevDmOHTuG+fPno0OHDjAwMBCSsbLGjh2L/Px8dOvWDSoqKpg5c2a5exXF2Lt3L2bNmiVMGjdo0CBs3rxZrs7Nmzflzs2ff/4ZwcHBGD16NJ49ewZLS0usXLkSU6dOFerEx8cjKytL6Wn6FSWRKTv5e01svJIxl2UvBjKZDEuXLsW2bduQmZmJ7t2749tvvxUmpABeT+M5f/587Nu3D/n5+ejbty8iIiJgbm6uUBzZ2dnQ1dUVpt8kUkT4iVvV1pnbr+KbUGtafYrlrZ1arVg9t+DajYOonikoKEBKSgpatmyp0FAoql+OHTuGoKAgXL9+vdrZCql6+fn5MDAwwLFjxyoceveh8vf3x/Pnz3H48OG6DqVOffLJJ+jUqRO++OKLSutUdU1VNDeo86F91ZFIJAgNDa0yk27SpAk2bdqETZs21WB0RERERDWjf//+uH37Nh4+fKjwD71UuZiYGPTp04dJFJVTWFiIDh06yM38WFvqxWQTRERERO+7ss8jqk67du2ER728adu2bRg9enRNhdUglU4Z/qEpfR5bRY4fP/4OI6m/1NXVyw1JrC1MpIiIiIjqmWPHjslNx11WZVND0/uv7MyCb2revDl69er17oIhJlJERERE9U1VDyquTzLyMhSqZ6yp+ENaqXJVTadO7x7vdiQiImpA6nCOKCKi90ZNXEuZSBERETUAjRs3BgDk5eXVcSRERA1f6bW09NqqDA7tIyIiagBUVFSgp6eHjIzXQ6k0NTUrfYwI0bvysrDi+7jeVNCooJYjIVKMTCZDXl4eMjIyoKenBxUVFaXbYiJFRETUQEilUgAQkimiuvai6IVC9bLVsms5EiJx9PT0hGuqsphIERERNRASiQSmpqYwNjaudEY3ondpX/I+her5tfSr5UiIFNe4ceO36okqxUSKiIiogVFRUamRLwFEb6tAotiQvSZNmtRyJETvHiebICIiIiIiEomJFBERERERkUhMpIiIiIiIiERiIkVERERERCQSEykiIiIiIiKRmEgRERERERGJxESKiIiIiIhIJCZSREREREREIjGRIiIiIiIiEomJFBERERERkUhMpIiIiIiIiERSresAiIiIiIjEikiKULjutI7TajES+lCxR4qIiIiIiEgk9kgRERERfQAU7cFh7w2RYphIERGdWq1YPbfg2o2DiIiIGgwO7SMiIiIiIhKJiRQREREREZFITKSIiIiIiIhEYiJFREREREQkEhMpIiIiIiIikZhIERERERERicTpz4lIYeEnblVbZ26/Vu8gEiIiIqK6xR4pIiIiIiIikZhIERERERERicREioiIiIiISCQmUkRERERERCIxkSIiIiIiIhKJiRQREREREZFITKSIiIiIiIhEYiJFREREREQkEhMpIiIiIiIikZhIERERERERicREioiIiIiISCQmUkRERERERCIxkSIiIiIiIhKJiRQREREREZFITKSIiIiIiIhEYiJFREREREQkEhMpIiIiIiIikWokkcrOzsbhw4eRnJxcE80RERERERHVa0olUiNGjMDmzZsBAPn5+XB0dMSIESPQvn17HDx4sEYDJCIiIiIiqm9UlVnpzJkzWLRoEQAgMjISMpkMz58/x+7du7FixQoMGzasRoMkojpyarXc2x73n1ZYLcEi4F1EQ0RERFRvKNUjlZWVBQMDAwBAVFQUhg0bBk1NTQwYMAC3b99WuJ0zZ85g4MCBMDMzg0QiweHDh+XK/f39IZFI5F49evSQq1NYWIiZM2fCyMgIWlpaGDRoEB48eKDMbhERERERESlEqUTK3Nwc8fHxyM3NRVRUFDw8PAAAmZmZaNKkicLt5ObmokOHDsIwwYp4eXkhLS1NeB07dkyufM6cOYiMjMTPP/+Ms2fPIicnBz4+PiguLlZm14iIiIiIiKql1NC+OXPmYPTo0dDW1oaFhQVcXV0BvO5hcnBwULgdb29veHt7V1lHXV0dUqm0wrKsrCxs374de/bsgbu7OwDgp59+grm5OX7//Xd4enoqHAsREREREZGilOqRmjZtGuLj47Fjxw6cO3cOjRq9buajjz7CihUrajTA06dPw9jYGK1atcLkyZORkZEhlF26dAkvX74UesQAwMzMDPb29oiLi6u0zcLCQmRnZ8u9iIiIiIiIFKX09OeOjo4YMGAAHj58iFevXgEABgwYgJ49e9ZYcN7e3ti7dy+io6MRFhaGxMRE9OnTB4WFhQCA9PR0qKmpQV9fX249ExMTpKenV9ru6tWroaurK7zMzc1rLGYiIiIiInr/KZVI5eXlYeLEidDU1ES7du1w//59AMCsWbOwZs2aGgtu5MiRGDBgAOzt7TFw4EAcP34ct27dwr///e8q15PJZJBIJJWWBwcHIysrS3ilpqbWWMxERERERPT+UyqRCg4OxpUrV3D69Gm5ySXc3d1x4MCBGgvuTaamprC0tBRmBpRKpSgqKkJmZqZcvYyMDJiYmFTajrq6Opo2bSr3IiIiIiIiUpRSidThw4exefNmfPzxx3I9P23btsWff/5ZY8G96enTp0hNTYWpqSkAoEuXLmjcuDFOnDgh1ElLS8P169fh7Oxca3EQEREREdGHTalZ+548eQJjY+Nyy3Nzc6scUvemnJwc3LlzR3ifkpKCpKQkGBgYwMDAAKGhoRg2bBhMTU1x9+5dfPHFFzAyMsLQoUMBALq6upg4cSLmzZsHQ0NDGBgYICgoCA4ODsIsfkRERERERDVNqR6prl27yt2nVJo8ff/993ByclK4nYsXL6JTp07o1KkTACAwMBCdOnXCkiVLoKKigmvXrmHw4MFo1aoVxo0bh1atWiE+Ph46OjpCG+Hh4RgyZAhGjBiBnj17QlNTE//617+goqKizK4RERERERFVS6keqdWrV8PLywv/+9//8OrVK2zcuBE3btxAfHw8YmJiFG7H1dUVMpms0vL//Oc/1bbRpEkTbNq0CZs2bVJ4u0RERERERG9DqR4pZ2dnnDt3Dnl5ebC2tsZvv/0GExMTxMfHo0uXLjUdIxERERERUb2iVI8UADg4OGD37t01GQsREREREVGDoHAilZ2dLUwTnp2dXWVdTidORERERETvM4UTKX19faSlpcHY2Bh6enoVzs5X+iDc4uLiGg2SiIiIiIioPlE4kYqOjoaBgQEA4NSpU7UWEBHRB+HUasXquQXXbhxERESkFIUTKRcXlwr/TURERERE9KFRata+nTt34pdffim3/JdffuEEFERERERE9N5TKpFas2YNjIyMyi03NjbGqlWr3jooIiIiIiKi+kypROrevXto2bJlueWWlpa4f//+WwdFRERERERUnymVSBkbG+Pq1avlll+5cgWGhoZvHRQREREREVF9ptQDeT/99FPMmjULOjo66N27NwAgJiYGs2fPxqefflqjARIRERER1TcRSREK153WcVotRkJ1RalEasWKFbh37x769u0LVdXXTZSUlGDs2LG8R4qIiIiIiN57SiVSampqOHDgAJYvX44rV65AQ0MDDg4OsLS0rOn4iIiIiIiI6h2lEqlSrVq1QqtWrWoqFiIiIiIiogZBqUSquLgYu3btwsmTJ5GRkYGSkhK58ujo6BoJjkis8BO3qq0ztx+TfyIiIiJ6O0olUrNnz8auXbswYMAA2NvbQyKR1HRcRERERERE9ZZSidTPP/+Mf/zjH+jfv39Nx0NERERERFTvKfUcKTU1NdjY2NR0LERERERERA2CUonUvHnzsHHjRshkspqOh4iIiIiIqN5Tamjf2bNncerUKRw/fhzt2rVD48aN5coPHTpUI8ERERERERHVR0olUnp6ehg6dGhNx0JERERERNQgKJVI7dy5s6bjICIiIiIiajCUukcKAF69eoXff/8d27Ztw4sXLwAAjx49Qk5OTo0FR0REREREVB8p1SN17949eHl54f79+ygsLES/fv2go6ODtWvXoqCgAFu3bq3pOImIiIiIiOoNpXqkZs+eDUdHR2RmZkJDQ0NYPnToUJw8ebLGgiMiIiIiIqqPlJ6179y5c1BTU5NbbmlpiYcPH9ZIYERERERERPWVUj1SJSUlKC4uLrf8wYMH0NHReeugiIiIiIiI6jOlEql+/fphw4YNwnuJRIKcnByEhISgf//+NRUbERERERFRvaTU0L7w8HC4ubmhbdu2KCgogJ+fH27fvg0jIyPs37+/pmMkIiIiIiKqV5RKpMzMzJCUlIT9+/fj8uXLKCkpwcSJEzF69Gi5ySeIiIiIiIjeR0olUgCgoaGBCRMmYMKECTUZDxERERERUb2nVCL1448/Vlk+duxYpYIhIiIiIiJqCJRKpGbPni33/uXLl8jLy4Oamho0NTWZSBERERER0XtNqVn7MjMz5V45OTm4efMmPv74Y042QURERERE7z2lEqmK2NraYs2aNeV6q4iIiIiIiN43NZZIAYCKigoePXpUk00SERERERHVO0rdI3XkyBG59zKZDGlpadi8eTN69uxZI4ERERERERHVV0olUkOGDJF7L5FI0KxZM/Tp0wdhYWE1ERcREREREVG9pVQiVVJSUtNxEBERERERNRg1eo8UERERERHRh0CpHqnAwECF665fv16ZTRAREREREdVbSiVS//3vf3H58mW8evUKrVu3BgDcunULKioq6Ny5s1BPIpHUTJRERERERET1iFKJ1MCBA6Gjo4Pdu3dDX18fwOuH9I4fPx69evXCvHnzajRIIiIiIiKi+kSpRCosLAy//fabkEQBgL6+PlasWAEPDw8mUkT01uL/elplecKrW5jbr9U7ioaI6O1FJEUoVG9ax2m1HAkR1QSlJpvIzs7G48ePyy3PyMjAixcv3jooIiIiIiKi+kypRGro0KEYP348fv31Vzx48AAPHjzAr7/+iokTJ8LX17emYyQiIiIiIqpXlBrat3XrVgQFBeGzzz7Dy5cvXzekqoqJEydi3bp1NRogERERERFRfaNUIqWpqYmIiAisW7cOf/75J2QyGWxsbKClpVXT8REREREREdU7b/VA3rS0NKSlpaFVq1bQ0tKCTCarqbiIiIiIiIjqLaUSqadPn6Jv375o1aoV+vfvj7S0NADApEmTOGMfERERERG995RKpObOnYvGjRvj/v370NTUFJaPHDkSUVFRNRYcERERERFRfaRUIvXbb7/hq6++QosWLeSW29ra4t69ewq3c+bMGQwcOBBmZmaQSCQ4fPiwXLlMJkNoaCjMzMygoaEBV1dX3LhxQ65OYWEhZs6cCSMjI2hpaWHQoEF48OCBMrtFRERERESkEKUSqdzcXLmeqFJ///031NXVRbXToUMHbN68ucLytWvXYv369di8eTMSExMhlUrRr18/uWdVzZkzB5GRkfj5559x9uxZ5OTkwMfHB8XFxeJ3jIiIiIiISAFKJVK9e/fGjz/+KLyXSCQoKSnBunXr4ObmpnA73t7eWLFiRYXPnpLJZNiwYQMWLVoEX19f2NvbY/fu3cjLy8O+ffsAAFlZWdi+fTvCwsLg7u6OTp064aeffsK1a9fw+++/V7rdwsJCZGdny72IiIiIiIgUpVQitW7dOmzbtg3e3t4oKirC559/Dnt7e5w5cwZfffVVjQSWkpKC9PR0eHh4CMvU1dXh4uKCuLg4AMClS5fw8uVLuTpmZmawt7cX6lRk9erV0NXVFV7m5uY1EjMREREREX0YlEqk2rZti6tXr6Jbt27o168fcnNz4evri//+97+wtraukcDS09MBACYmJnLLTUxMhLL09HSoqalBX1+/0joVCQ4ORlZWlvBKTU2tkZiJiIiIiOjDIPqBvKU9QNu2bcPSpUtrIyY5EolE7r1MJiu37E3V1VFXVxd1LxcREREREVFZohOpxo0b4/r169UmM29LKpUCeN3rZGpqKizPyMgQeqmkUimKioqQmZkp1yuVkZEBZ2fnWo2PiKjBObVasXpuwbUbBxER0XtAqaF9Y8eOxfbt22s6FjktW7aEVCrFiRMnhGVFRUWIiYkRkqQuXbqgcePGcnXS0tJw/fp1JlJERERERFRrRPdIAa8Tmh9++AEnTpyAo6MjtLS05MrXr1+vUDs5OTm4c+eO8D4lJQVJSUkwMDCAhYUF5syZg1WrVsHW1ha2trZYtWoVNDU14efnBwDQ1dXFxIkTMW/ePBgaGsLAwABBQUFwcHCAu7u7MrtGRERERERULVGJ1F9//QUrKytcv34dnTt3BgDcunVLro6YIX8XL16Umy49MDAQADBu3Djs2rULn3/+OfLz8zFt2jRkZmaie/fu+O2336CjoyOsEx4eDlVVVYwYMQL5+fno27cvdu3aBRUVFTG7RkREREREpDBRiZStrS3S0tJw6tQpAMDIkSPxzTfflJtZT1Gurq6QyWSVlkskEoSGhiI0NLTSOk2aNMGmTZuwadMmpWIgIiIiIiISS9Q9Um8mPcePH0dubm6NBkRERERERFTfKTXZRKmqepOIiIiIiIjeV6KG9kkkknL3QNX2NOhE9P+rYurqHvefCv9OsAh4F9EQERERfdBEJVIymQz+/v7Cw2wLCgowderUcrP2HTp0qOYiJCIiIiIiqmdEJVLjxo2Te//ZZ5/VaDBEREREREQNgahEaufOnbUVBxERERERUYPxVpNNEBERERERfYiYSBEREREREYnERIqIiIiIiEgkJlJEREREREQiMZEiIiIiIiISiYkUERERERGRSEykiIiIiIiIRGIiRUREREREJBITKSIiIiIiIpGYSBEREREREYnERIqIiIiIiEgkJlJEREREREQiMZEiIiIiIiISiYkUERERERGRSEykiIiIiIiIRGIiRUREREREJBITKSIiIiIiIpGYSBEREREREYnERIqIiIiIiEgkJlJEREREREQiMZEiIiIiIiISiYkUERERERGRSKp1HQB9GMJP3Kq2ztx+rd5BJEREREREb489UkRERERERCIxkSIiIiIiIhKJiRQREREREZFIvEeKiIiIiKgeiEiKULjutI7TajESUgR7pIiIiIiIiERiIkVERERERCQSEykiIiIiIiKRmEgRERERERGJxESKiIiIiIhIJM7aR0RERB8EzohGRDWJPVJEREREREQiMZEiIiIiIiISiYkUERERERGRSEykiIiIiIiIRGIiRUREREREJBITKSIiIiIiIpGYSBEREREREYnERIqIiIiIiEgkJlJEREREREQiMZEiIiIiIiISiYkUERERERGRSPU6kQoNDYVEIpF7SaVSoVwmkyE0NBRmZmbQ0NCAq6srbty4UYcRExERERHRh6BeJ1IA0K5dO6SlpQmva9euCWVr167F+vXrsXnzZiQmJkIqlaJfv3548eJFHUZMRERERETvu3qfSKmqqkIqlQqvZs2aAXjdG7VhwwYsWrQIvr6+sLe3x+7du5GXl4d9+/bVcdRERERERPQ+q/eJ1O3bt2FmZoaWLVvi008/xV9//QUASElJQXp6Ojw8PIS66urqcHFxQVxcXJVtFhYWIjs7W+5FRERERESkqHqdSHXv3h0//vgj/vOf/+D7779Heno6nJ2d8fTpU6SnpwMATExM5NYxMTERyiqzevVq6OrqCi9zc/Na2wciIiIiInr/1OtEytvbG8OGDYODgwPc3d3x73//GwCwe/duoY5EIpFbRyaTlVv2puDgYGRlZQmv1NTUmg+eiIiIiIjeW/U6kXqTlpYWHBwccPv2bWH2vjd7nzIyMsr1Ur1JXV0dTZs2lXsREREREREpqkElUoWFhUhOToapqSlatmwJqVSKEydOCOVFRUWIiYmBs7NzHUZJRERERETvO9W6DqAqQUFBGDhwICwsLJCRkYEVK1YgOzsb48aNg0QiwZw5c7Bq1SrY2trC1tYWq1atgqamJvz8/Oo6dCIiIiIieo/V60TqwYMHGDVqFP7++280a9YMPXr0QEJCAiwtLQEAn3/+OfLz8zFt2jRkZmaie/fu+O2336Cjo1PHkRMRERER0fusXidSP//8c5XlEokEoaGhCA0NfTcBEZU6tRo97j9VoOLXtR4KEREREb17DeoeKSIiIiIiovqAiRQREREREZFITKSIiIiIiIhEYiJFREREREQkEhMpIiIiIiIiker1rH1ERNTAnVqtWD234NqNg4iIqIaxR4qIiIiIiEgkJlJEREREREQicWgfEX3Qwk/cqvbhyk4fGb6jaIiIiKihYI8UERERERGRSEykiIiIiIiIRGIiRUREREREJBITKSIiIiIiIpE42QQRERHVOxFJEQrXndZxWi1GQkRUMfZIERERERERicREioiIiIiISCQmUkRERERERCIxkSIiIiIiIhKJiRQREREREZFITKSIiIiIiIhEYiJFREREREQkEhMpIiIiIiIikZhIERERERERicREioiIiIiISCQmUkRERERERCKp1nUAVLfCT9yqsnxuv1bvKBIiIiIiooaDPVJEREREREQiMZEiIiIiIiISiYkUERERERGRSLxHihq+U6uFf/a4/7TSagkWAe8iGiIiIiL6ALBHioiIiIiISCQmUkRERERERCIxkSIiIiIiIhKJiRQREREREZFInGyCiIiIyolIilC47rSO02oxEiKi+omJFBERERER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",
            "text/plain": [
              "<Figure size 1000x400 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "viz_distribusi_binomial(first_binomial, second_binomial, third_binomial)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7OQcWau-iE6V"
      },
      "source": [
        "##### Output yang diharapkan\n",
        "\n",
        "<img src=\"https://drive.google.com/uc?id=1Vq88KSfdkbjT_R0PaChjZwHnSxVHcMdI\" style=\"height:300px;\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-a5kiWaSiqQs"
      },
      "source": [
        "# Segmen 2: Implementasi Algoritma Naive Bayes dengan Data Sintetis\n",
        "\n",
        "Selamat! Anda telah mempelajari dasar-dasar distribusi probabilitas pada segmen pertama. Kali ini, Anda akan diajak untuk mengintegrasikan teori distribusi probabilitas dengan algoritma Naive Bayes.\n",
        "\n",
        "Segmen ini akan berfokus pada dua hal:\n",
        "\n",
        "* Membuat data sintetis yang mengikuti distribusi tertentu.\n",
        "\n",
        "* Melakukan klasifikasi Naive Bayes terhadap data sintetis yang telah dibuat.\n",
        "\n",
        "Pada tahap klasifikasi, Naive Bayes akan digunakan untuk mengklasifikasikan **jenis burung** berdasarkan karakteristiknya."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uO52K8V1Xhlv"
      },
      "source": [
        "## Segmen 2.1: Membuat Data Sintetis\n",
        "\n",
        "Pada tahap ini, Anda akan membuat dataset sintetis yang merepresentasikan jenis burung berdasarkan empat karakteristik atau atribut berikut:\n",
        "\n",
        "* **wingspan_cm**: Ukuran bentang sayap burung dalam sentimeter, mengikuti distribusi Gaussian.\n",
        "* **weight_g**: Berat burung dalam gram, mengikuti distribusi Gaussian.\n",
        "* **sing_days**: Jumlah hari burung berkicau dalam sebulan (dengan asumsi satu bulan = 30 hari), mengikuti distribusi binomial.\n",
        "* **beak_head_ratio**: Rasio panjang paruh hingga kepala burung, mengikuti distribusi uniform.\n",
        "\n",
        "Karena sebelumnya sudah membuat fungsi untuk menghasilkan berbagai distribusi tersebut, Anda akan memanfaatkannya pada tahap ini."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "id": "kuC4Uwqzisgy"
      },
      "outputs": [],
      "source": [
        "# Mendefinisikan nama-nama kolom/fitur dataset\n",
        "FEATURES = [\"wingspan_cm\", \"weight_g\", \"sing_days\", \"beak_head_ratio\"]"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "D4HgYgDTatLx"
      },
      "source": [
        "Selanjutnya, Anda perlu menyimpan parameter distribusi untuk setiap fiturnya. Misalnya, fitur `weight_g` memiliki distribusi Gaussian dan parameter miu dan sigma.\n",
        "\n",
        "Agar meminimalkan kompleksitas kode, kita akan menggunakan [dataclass](https://docs.python.org/3/library/dataclasses.html) pada setiap parameter untuk meminimalkan kompleksitas kode.\n",
        "\n",
        "Jangan khawatir jika Anda belum familier dengan fungsi ini. dataclass adalah decorator pada Python (dikenalkan sejak Python 3.7) yang secara otomatis menambahkan *boilerplate code* dalam sebuah class untuk menyimpan data.\n",
        "\n",
        "Contohnya, Anda memiliki class berikut.\n",
        "\n",
        "```\n",
        "@dataclass\n",
        "class my_course:\n",
        "  course_id: int\n",
        "  course_name: string\n",
        "\n",
        "foo = my_course(course_name='Dicoding')\n",
        "```\n",
        "\n",
        "Anda dapat mengakses informasi `course_name` dari `foo` melalui sintaks `foo.course_name` yang akan mengembalikan string \"Dicoding\".\n",
        "\n",
        "\n",
        "> Mengapa menggunakan dataclass? Anda perlu menyimpan parameter distribusi untuk setiap fitur. Misalnya, untuk weight dan height, diperlukan parameter μ dan σ. Jika dibuat terpisah, seperti miu_weight. sigma_weight, miu_height, sigma_height, akan tidak efektif karena terlalu panjang dan beragam.\n",
        "\n",
        "__repr__ adalah method yang digunakan untuk menampilkan objek saat Anda print.\n",
        "\n",
        "Tanpa `__repr__`:\n",
        "```\n",
        "<my_class object at 0x7f9f3b4d>\n",
        "```\n",
        "\n",
        "Dengan `__repr__`:\n",
        "```\n",
        "my_class(a='Dicoding')\n",
        "```"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "q6mEKNGvFUpz"
      },
      "outputs": [],
      "source": [
        "# Mendefinisikan parameter distribusi dalam class.\n",
        "\n",
        "@dataclass\n",
        "class gaussian_params:\n",
        "  miu: float\n",
        "  sigma: float\n",
        "\n",
        "  def __repr__(self):\n",
        "    return f\"gaussian_params(mu={self.miu:.3f}, sigma={self.sigma:.3f})\"\n",
        "\n",
        "@dataclass\n",
        "class binomial_params:\n",
        "  n_trials: int\n",
        "  probability: float\n",
        "\n",
        "  def __repr__(self):\n",
        "    return f\"binomial_params(n_trials={self.n_trials:.3f}, probability={self.probability:.3f})\"\n",
        "\n",
        "@dataclass\n",
        "class uniform_params:\n",
        "  lower_bound: int\n",
        "  upper_bound: int\n",
        "\n",
        "  def __repr__(self):\n",
        "    return f\"uniform_params(lower_bound={self.lower_bound:.3f}, upper_bound={self.upper_bound:.3f})\""
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1O9HsKz8KrEs"
      },
      "source": [
        "Setelah mendefinisikan *class* untuk menyimpan parameter distribusi, sekarang kita perlu membuat sebuah dictionary bernama `breed_params` yang berisi semua informasi distribusi untuk masing-masing jenis burung.\n",
        "\n",
        "Dalam dictionary ini, setiap key mewakili satu jenis burung (dengan label numerik 0,1, dan 2) dan setiap value berisi parameter distribusi untuk semua atribut fitur (seperti wingspan, weight, sing_days, dan beak_head_ratio).\n",
        "\n",
        "Nantinya, dictionary ini digunakan untuk menghasilkan data sintetis berdasarkan jenis distribusi dan parameter yang sudah ditetapkan."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "Vv0MB5q_HDoJ"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "{0: {'wingspan_cm': gaussian_params(mu=35.000, sigma=1.500),\n",
              "  'weight_g': gaussian_params(mu=20.000, sigma=1.000),\n",
              "  'sing_days': binomial_params(n_trials=30.000, probability=0.800),\n",
              "  'beak_head_ratio': uniform_params(lower_bound=0.600, upper_bound=0.100)},\n",
              " 1: {'wingspan_cm': gaussian_params(mu=30.000, sigma=2.000),\n",
              "  'weight_g': gaussian_params(mu=25.000, sigma=5.000),\n",
              "  'sing_days': binomial_params(n_trials=30.000, probability=0.500),\n",
              "  'beak_head_ratio': uniform_params(lower_bound=0.200, upper_bound=0.500)},\n",
              " 2: {'wingspan_cm': gaussian_params(mu=40.000, sigma=3.500),\n",
              "  'weight_g': gaussian_params(mu=32.000, sigma=3.000),\n",
              "  'sing_days': binomial_params(n_trials=30.000, probability=0.300),\n",
              "  'beak_head_ratio': uniform_params(lower_bound=0.100, upper_bound=0.300)}}"
            ]
          },
          "execution_count": 22,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "breed_params  = {\n",
        "    0: {\n",
        "        \"wingspan_cm\": gaussian_params(miu=35, sigma=1.5),\n",
        "        \"weight_g\": gaussian_params(miu=20, sigma=1),\n",
        "        \"sing_days\": binomial_params(n_trials=30, probability=0.8),\n",
        "        \"beak_head_ratio\": uniform_params(lower_bound=0.6, upper_bound=0.1)\n",
        "    },\n",
        "    1: {\n",
        "        \"wingspan_cm\": gaussian_params(miu=30, sigma=2),\n",
        "        \"weight_g\": gaussian_params(miu=25, sigma=5),\n",
        "        \"sing_days\": binomial_params(n_trials=30, probability=0.5),\n",
        "        \"beak_head_ratio\": uniform_params(lower_bound=0.2, upper_bound=0.5)\n",
        "    },\n",
        "    2: {\n",
        "        \"wingspan_cm\": gaussian_params(miu=40, sigma=3.5),\n",
        "        \"weight_g\": gaussian_params(miu=32, sigma=3),\n",
        "        \"sing_days\": binomial_params(n_trials=30, probability=0.3),\n",
        "        \"beak_head_ratio\": uniform_params(lower_bound=0.1, upper_bound=0.3)\n",
        "    }\n",
        "}\n",
        "\n",
        "breed_params"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "3rxLC5-u-UJL"
      },
      "source": [
        "### Tugas 1: Menghasilkan Data Sintetis\n",
        "\n",
        "*Mantap!* Kita sudah menyiapkan parameter setiap distribusi beserta jenis burungnya (direpresentasikan oleh numerik 0, 1, dan 2).\n",
        "\n",
        "Sekarang, Anda perlu membuat sebuah fungsi yang ditujukan untuk membuat data sintetis berdasarkan parameter-parameter tersebut.\n",
        "\n",
        "Tugasnya sederhana, Anda perlu mengisi bagian antara `# MULAI KODE DI SINI` dan `# AKHIRI KODE DI SINI` dengan petunjuk berikut.\n",
        "\n",
        "* Melakukan `loop` untuk setiap fitur yang tersimpan pada variabel `FEATURES`.\n",
        "* Periksa nama fitur untuk menentukan jenis distribusi yang sesuai.\n",
        "* Menghasilkan data acak sesuai distribusi dengan memanggil fungsi yang sudah Anda buat sebelumnya.\n",
        "  * `generate_gaussian` untuk fitur yang distribusinya Gaussian.\n",
        "  * `generate_binomial` untuk fitur yang distribusinya binomial.\n",
        "  * `generate_rand_uniform` untuk fitur yang distribusinya uniform.\n",
        "\n",
        "<details>\n",
        "<summary>\n",
        "<font color=\"yellow\">PETUNJUK!</font>\n",
        "</summary>\n",
        "\n",
        "Gunakan parameter dari variabel `breed_params` ketika memanggil fungsi distribusi, seperti `generate_gaussian`.\n",
        "\n",
        "Contohnya, untuk fitur `wingspan_cm` pada breed 0, bisa kamu tulis seperti berikut.\n",
        "```\n",
        "generate_gaussian(breed_params[0]['wingspan_cm'].miu, breed_params[0]['wingspan_cm'].sigma, n_samples)\n",
        "```"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "FFoPlIbbHQLj"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>wingspan_cm</th>\n",
              "      <th>weight_g</th>\n",
              "      <th>sing_days</th>\n",
              "      <th>beak_head_ratio</th>\n",
              "      <th>breed</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1451</th>\n",
              "      <td>28.716498</td>\n",
              "      <td>21.791245</td>\n",
              "      <td>13.0</td>\n",
              "      <td>0.278155</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>51</th>\n",
              "      <td>34.339161</td>\n",
              "      <td>19.559440</td>\n",
              "      <td>23.0</td>\n",
              "      <td>0.435117</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>567</th>\n",
              "      <td>34.706978</td>\n",
              "      <td>19.804652</td>\n",
              "      <td>24.0</td>\n",
              "      <td>0.388720</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1612</th>\n",
              "      <td>28.209610</td>\n",
              "      <td>20.524025</td>\n",
              "      <td>13.0</td>\n",
              "      <td>0.255602</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1164</th>\n",
              "      <td>33.875004</td>\n",
              "      <td>19.250003</td>\n",
              "      <td>22.0</td>\n",
              "      <td>0.486686</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1094</th>\n",
              "      <td>36.660672</td>\n",
              "      <td>21.107115</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0.167061</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>229</th>\n",
              "      <td>35.033846</td>\n",
              "      <td>20.022564</td>\n",
              "      <td>24.0</td>\n",
              "      <td>0.345499</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>298</th>\n",
              "      <td>36.118026</td>\n",
              "      <td>20.745350</td>\n",
              "      <td>26.0</td>\n",
              "      <td>0.214015</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>605</th>\n",
              "      <td>35.673931</td>\n",
              "      <td>20.449287</td>\n",
              "      <td>25.0</td>\n",
              "      <td>0.263306</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2227</th>\n",
              "      <td>30.777602</td>\n",
              "      <td>26.944005</td>\n",
              "      <td>16.0</td>\n",
              "      <td>0.395386</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "      wingspan_cm   weight_g  sing_days  beak_head_ratio  breed\n",
              "1451    28.716498  21.791245       13.0         0.278155      1\n",
              "51      34.339161  19.559440       23.0         0.435117      0\n",
              "567     34.706978  19.804652       24.0         0.388720      0\n",
              "1612    28.209610  20.524025       13.0         0.255602      1\n",
              "1164    33.875004  19.250003       22.0         0.486686      0\n",
              "1094    36.660672  21.107115       26.0         0.167061      0\n",
              "229     35.033846  20.022564       24.0         0.345499      0\n",
              "298     36.118026  20.745350       26.0         0.214015      0\n",
              "605     35.673931  20.449287       25.0         0.263306      0\n",
              "2227    30.777602  26.944005       16.0         0.395386      1"
            ]
          },
          "execution_count": 23,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "def generate_data_synthetic(breed, features, n_samples, params):\n",
        "  \"\"\"\n",
        "  Menghasilkan data sintetis untuk jenis burung secara spesifik berdasarkan fitur dan parameter yang diberikan.\n",
        "\n",
        "  Parameters:\n",
        "  - breed (str): Jenis burung untuk data yang dihasilkan nantinya.\n",
        "  - freatures (list[str]): List dari fitur setiap data. Misalnya ['wingspan_cm', 'weight_g', 'sing_days', 'beak_head_ratio']\n",
        "  - n_samples (int): Banyaknya sampel yang akan dihasilkan.\n",
        "  - params (dict): Dictionary yang terdiri dari parameter untuk setiap jenis burung.\n",
        "\n",
        "  Returns:\n",
        "  - df (pandas.DataFrame): Sebuah dataframe yang berisikan data sintetis jenis burung.\n",
        "  \"\"\"\n",
        "  df = pd.DataFrame()\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "\n",
        "  # Lakukan loop untuk setiap fitur dalam list features\n",
        "\n",
        "\n",
        "      # Gunakan pernyataan match-case (atau bisa juga if-else) untuk memilih distribusi yang sesuai berdasarkan nama fitur\n",
        "      # Anda dapat mencocokkan nama fitur dengan beberapa kemungkinan nilai\n",
        "\n",
        "\n",
        "              # Untuk fitur \"wingspan_cm\" dan \"weight_g\" dengan distribusi Gaussian\n",
        "              # Gunakan fungsi generate_gaussian dengan parameter mean dan standard deviation\n",
        "              # Simpan hasil generasi pada kolom dataframe sesuai nama fitur\n",
        "\n",
        "\n",
        "\n",
        "              # Untuk fitur \"sing_days\" dengan distribusi binomial\n",
        "              # Gunakan fungsi generate_binomial dengan parameter n_trials dan probability\n",
        "              # Simpan hasil generasi pada kolom dataframe sesuai dengan nama fitur\n",
        "\n",
        "\n",
        "\n",
        "              # Untuk fitur \"beak_head_ratio\" dengan distribusi Uniform\n",
        "              # Gunakan fungsi generate_rand_uniform dengan parameter lower_bound dan upper_bound\n",
        "              # Simpan hasil generasi pada kolom dataframe sesuai dengan nama fitur\n",
        "  for feature in features:\n",
        "    # Distribusi Gaussian\n",
        "    if feature in [\"wingspan_cm\", \"weight_g\"]:\n",
        "        mu = params[breed][feature].miu\n",
        "        sigma = params[breed][feature].sigma\n",
        "        df[feature] = generate_gaussian(mu, sigma, n_samples)\n",
        "\n",
        "    # Distribusi Binomial\n",
        "    elif feature == \"sing_days\":\n",
        "        n_trials = params[breed][feature].n_trials\n",
        "        probability = params[breed][feature].probability\n",
        "        df[feature] = generate_binomial(n_trials, probability, n_samples)\n",
        "\n",
        "    # Distribusi Uniform\n",
        "    elif feature == \"beak_head_ratio\":\n",
        "        lower_bound = params[breed][feature].lower_bound\n",
        "        upper_bound = params[breed][feature].upper_bound\n",
        "        df[feature] = generate_rand_uniform(lower_bound, upper_bound, n_samples)\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  df['breed'] = breed\n",
        "\n",
        "  return df\n",
        "\n",
        "# Generate data for each breed\n",
        "df_0 = generate_data_synthetic(breed=0, features=FEATURES, n_samples=1200, params=breed_params)\n",
        "df_1 = generate_data_synthetic(breed=1, features=FEATURES, n_samples=1350, params=breed_params)\n",
        "df_2 = generate_data_synthetic(breed=2, features=FEATURES, n_samples=900, params=breed_params)\n",
        "\n",
        "# Concatenate all breeds into a single dataframe\n",
        "df_all_breeds = pd.concat([df_0, df_1, df_2]).reset_index(drop=True)\n",
        "\n",
        "# Shuffle the data\n",
        "df_all_breeds = df_all_breeds.sample(frac = 1, random_state=42)\n",
        "\n",
        "# Print the dataframe\n",
        "df_all_breeds.head(10)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UMduKeqzTN25"
      },
      "source": [
        "#### Output yang diharapkan\n",
        "<img src=\"https://drive.google.com/uc?id=1-3FgN4dbUq2Bh3o8GzqnrPKFmKydXle5\"\n",
        "style=\"height:300px;\"/>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4ffH6bb6UCwi"
      },
      "source": [
        "Hebat! Data sintetis berhasil dibuat!\n",
        "\n",
        "Sebelum memulai proses training, kita perlu membagi dataset menjadi data training dan data testing. Anda akan menggunakan 70% dari dataset untuk training dan 30% sisanya untuk testing."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "X0wLVznpJsOv"
      },
      "outputs": [],
      "source": [
        "split = int(len(df_all_breeds) * 0.7)\n",
        "\n",
        "df_train = df_all_breeds[:split].reset_index(drop=True)\n",
        "df_test = df_all_breeds[split:].reset_index(drop=True)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "q4S5rwMBUfdF"
      },
      "source": [
        "## Segmen 2.1: Mengimplementasikan Algoritma Naive Bayes"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DI4-cgujUm-a"
      },
      "source": [
        "Mari kita lakukan **rekap singkat** tentang algoritma Naive Bayes.\n",
        "\n",
        "Naive Bayes adalah algoritma yang sering digunakan dalam **teknik klasifikasi (supervised learning)** untuk menetapkan label kelas pada data berdasarkan atribut atau karakteristiknya dengan memanfaatkan **probabilitas bersyarat**.\n",
        "\n",
        "**Probabilitas bersyarat** sendiri adalah ukuran peluang terjadinya suatu peristiwa dengan syarat bahwa peristiwa lain sudah diketahui terjadi.\n",
        "\n",
        "Untuk memudahkan pemahaman, mari kita lihat contoh berikut.\n",
        "\n",
        "Misalkan kita memiliki \\$X\\$, yaitu sekumpulan data training.\n",
        "Setiap elemen \\$x \\in X\\$ (dibaca: **x adalah anggota dari himpunan X**) direpresentasikan sebagai sebuah **vektor** dengan bentuk\n",
        "\n",
        "$$\n",
        "x = (x_1, x_2, \\ldots, x_n)\n",
        "$$\n",
        "\n",
        "\\$n\\$ adalah jumlah atribut dalam setiap sampel.\n",
        "\n",
        "Sebagai contoh, \\$X\\$ bisa berupa data tentang 1000 ekor burung, yaitu setiap burung dideskripsikan dengan tiga atribut, seperti panjang sayap (wingspan), berat (weight), dan rasio paruh terhadap kepala (beak\\_head\\_ratio).\n",
        "\n",
        "Dengan demikian, himpunan \\$X\\$ ditulis sebagai berikut.\n",
        "\n",
        "$$\n",
        "X = \\{ \\text{bird}_1, \\text{bird}_2, \\ldots, \\text{bird}_{1000} \\}\n",
        "$$\n",
        "\n",
        "\\$\\text{bird}\\_1\\$ adalah salah satu anggota himpunan yang direpresentasikan sebagai vektor berdimensi 3:\n",
        "\n",
        "$$\n",
        "\\text{bird}_1 = (\\text{wingspan}_{\\text{bird}_1}, \\text{weight}_{\\text{bird}_1}, \\text{beak_head_ratio}_{\\text{bird}_1})\n",
        "$$\n",
        "\n",
        "---\n",
        "\n",
        "Sekarang, **tujuan utama kita** adalah memprediksi kelas atau jenis burung berdasarkan atribut-atribut tersebut.\n",
        "Misalnya kita memiliki \\$m\\$ kelas\n",
        "\n",
        "$$\n",
        "C_1, C_2, \\ldots, C_m\n",
        "$$\n",
        "\n",
        "Menggunakan contoh di atas, kita bisa memiliki \\$m = 3\\$ kelas untuk setiap jenis burung yang terdapat pada data training.\n",
        "Naive Bayes melakukan prediksi ini dengan cara menghitung **posterior probabilities** yang menyatakan seberapa besar kemungkinan sebuah sampel termasuk dalam kelas \\$C\\_i\\$, yaitu\n",
        "\n",
        "$$\n",
        "P(C_i \\mid x), \\quad i = 1, \\ldots, m.\n",
        "$$\n",
        "\n",
        "Kelas yang diprediksi adalah kelas \\$C\\_i\\$ dengan nilai probabilitas tertinggi.\n",
        "Secara lebih formal, dari semua nilai posterior probability untuk sampel tersebut, Naive Bayes memilih kelas dengan rumus berikut.\n",
        "\n",
        "$$\n",
        "\\text{Prediksi kelas untuk } x = \\arg \\max \\left\\{ P(C_1 \\mid x), P(C_2 \\mid x), \\ldots, P(C_m \\mid x) \\right\\}\n",
        "$$\n",
        "\n",
        "Contohnya, jika nilai tertinggi adalah \\$P(C\\_5 | x)\\$, maka\n",
        "\n",
        "$$\n",
        "\\arg \\max \\left\\{ P(C_1 \\mid x), P(C_2 \\mid x), \\ldots, P(C_m \\mid x) \\right\\} = 5\n",
        "$$\n",
        "\n",
        "---\n",
        "\n",
        "Lalu, **apa sebenarnya yang dimaksud dengan posterior probability? Bagaimana cara menghitungnya?**\n",
        "\n",
        "**Posterior probability** adalah probabilitas bahwa sebuah hipotesis atau kelas tertentu benar setelah kita mempertimbangkan data yang kita amati.\n",
        "Sederhananya, ini seperti “tingkat keyakinan” kita pada suatu kelas setelah melihat ciri-ciri datanya.\n",
        "\n",
        "Nilai ini dapat dihitung menggunakan rumus berikut.\n",
        "\n",
        "$$\n",
        "P(C_i \\mid x) = \\frac{P(x \\mid C_i)P(C_i)}{P(x)}\n",
        "$$\n",
        "\n",
        " **Artinya**\n",
        "\n",
        "* $P(C_i \\mid x)$ adalah **posterior probability**, yaitu probabilitas sampel x termasuk pada kelas $C_i$ setelah mempertimbangkan atribut-atributnya.\n",
        "* $P(x \\mid C_i)$ adalah **likelihood**, yaitu probabilitas kita mengamati data x jika kita tahu sampel tersebut berasal dari kelas $C_i$.\n",
        "* $P(C_i)$ adalah **prior probability**, yaitu probabilitas awal atau dugaan awal bahwa sebuah sampel termasuk kelas $C_i$ sebelum melihat data.\n",
        "* $P(x)$ adalah **evidence** atau **marginal likelihood**, yaitu probabilitas keseluruhan untuk mengamati data x, terlepas dari kelasnya.\n",
        "* $C_i$ adalah **kelas ke-i**, salah satu dari semua kelas yang mungkin.\n",
        "* $x$ adalah **vektor fitur atau atribut** dari satu sampel data yang ingin kita klasifikasikan.\n",
        "\n",
        "Secara umum, mengihitung langsung $P(x \\mid C_i)$ untuk setiap kombinasi atribut dan kelas itu bisa rumit dan memakan banyak waktu. Untuk menyederhanakan perhitungan, algoritma Naive Bayes membuat asumsi \"naive\" atau sederhana, yaitu menganggap setiap atribut bersifat bebas satu sama lain dalam kelas yang sama.\n",
        "\n",
        "Artinya, kita menganggap nilai satu atribut tidak memengaruhi nilai atribut lain, jika sudah diketahui kelasnya.\n",
        "\n",
        "Misalnya, kita anggap bahwa dalam satu jenis burung tertentu, atribut seperti berat, lebar sayap, dan rasio paruh terhadap kepala tidak saling berkaitan atau memengaruhi satu sama lain.\n",
        "\n",
        "Berarti, kita mengasumsikan bahwa jika kita sudah tahu jenis burungnya, nilai berat tidak memberi petunjuk apa-apa tentang lebar sayap atau rasio paruh-kepala. Misalnya, burung itu bisa berbobot 200 gram dengan lebar sayap 30 cm, atau 200 gram dengan lebar sayap 35 cm—semuanya sama-sama mungkin tanpa aturan hubungan tertentu di antara atribut-atribut itu.\n",
        "\n",
        "---\n",
        "\n",
        "Dengan asumsi ini, kita tidak perlu menghitung semua kombinasi atribut dengan rumit. Jadi, rumus berikut didapatkan.\n",
        "\n",
        "$$P(x \\mid C_i) = P(x_1 \\mid C_i) \\cdot P(x_2 \\mid C_i) \\cdot \\ldots \\cdot P(x_n \\mid C_i) = \\prod_{k = 1}^{n} P(x_k \\mid C_i).$$\n",
        "\n",
        "Tujuan rumusnya menjadi sederhana, Anda hanya perlu mengalikan probabilitas masing-masing atribut secara terpisah.\n",
        "\n",
        "Probabilitas $P(x_k \\mid C_i)$ dapat diestimasi dari data training, tapi cara menghitungnya dapat berbeda tergantung tipe data atributnya.\n",
        "- Jika atribut kategorikal, $P(x_k \\mid C_i)$ dihitung sebagai frekuensi nilai itu pada kelas $C_i$ (Dijelaskan pada materi latihan di modul ke-2).\n",
        "- Jika atribut bernilai kontinu, Anda perlu menganggap nilai tersebut mengikuti distribusi tertentu (misalnya Gaussian) dan estimasi parameternya dari data training. Untuk distirbusi Gaussian, kita hitung $\\mu$ dan $\\sigma$ dari data training di kelas itu, lalu\n",
        "$$P(x_k \\mid C_i) = \\text{PDF}_{\\text{gaussian}}(x_k, \\mu_{C_i}, \\sigma_{C_i})$$\n",
        "\n",
        "> Mengapa harus menggunakan distribusi untuk nilai kontinu atau diskret? Sebab tidak ada nilai pasti suatu atribut bernilai $x$, seperti 50, 50.4, atau lainnya. Kita hanya bisa menghitung probabilitas angka tersebut melalui distribusinya.\n",
        "\n",
        "Inilah tahapan kita untuk menklasifikasikan jenis burung menggunakan algoritma Naive Bayes.\n",
        "1. Menghitung probability density function (PDF) untuk distribusi data.\n",
        "2. Mengestimasikan parameter.\n",
        "3. Menghitung probabilitas X adalah jenis burung tertentu (posterior probability) -> $P(x \\mid C_i)$.\n",
        "\n",
        "\n",
        "> Catatan: Anda dapat mempelajari kembali materi tentang Naive Bayes pada modul ke-2 kelas Matematika untuk Data Science."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "xtLhKPT--D-J"
      },
      "source": [
        "### Tugas 2: PDF untuk Distribusi Uniform\n",
        "\n",
        "Untuk menggunakan algoritma Naive Bayes dalam memprediksi kelas, kita perlu menghitung **probabilitas tiap kelas**. Salah satu langkah penting adalah menghitung prior probability, yaitu peluang awal sebuah sampel termasuk ke dalam kelas tertentu sebelum melihat datanya.\n",
        "\n",
        "Meskipun dengan data sintetis yang dihasilkan kita sudah tahu nilai prior (proporsi jenis/kelas burung dalam data sintetis), kita masih perlu cara untuk menghitung probabilitas fitur tertentu untuk sebuah kelas.\n",
        "\n",
        "Karena data kita memiliki tiga jenis distribusi, yaitu uniform, Gaussian, dan binomial, kita akan membuat probability density function (PDF) untuk masing-masing distribusi tersebut.\n",
        "\n",
        "Untuk rumus uniform PDF, jika sebuah variabel acak $X$ mengikuti distribusi $Uniform(a,b)$, PDF untuk $X$ memiliki rumus berikut.\n",
        "\n",
        "$$f(x;a,b) =\n",
        "\\begin{cases}\n",
        "\\frac{1}{b-a}, \\quad \\text{if } x \\in [a,b]. \\\\\n",
        "0, \\quad \\text{otherwise.}\n",
        "\\end{cases}\n",
        "$$\n",
        "\n",
        "Secara probabilitas, ini artinya:\n",
        "\n",
        "- Jika 𝑥 ∈[ 𝑎,𝑏], maka peluang relatifnya konstan (semua nilai dalam interval sama-sama mungkin) dan dapat dihitung dengan rumus $\\frac{1}{b-a}$.\n",
        "\n",
        "- Jika 𝑥 berada di luar interval [𝑎,𝑏] peluangnya nol.\n",
        "- 𝑎 adalah batas bawah, 𝑏 adalah batas atas.\n",
        "\n",
        "Dalam tugas selanjutnya, Anda diminta untuk membuat sebuah fungsi yang bisa menghasilkan pdf uniform berdasarkan rumus yang diberikan sebelumnya."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "id": "q41wZw8ELhYQ"
      },
      "outputs": [],
      "source": [
        "def pdf_for_uniform(val, lower_bound, upper_bound):\n",
        "  \"\"\"\n",
        "  Menghitung probability density function (PDF) untuk distribusi uniform antara `lower_bound` dan `upper_bound` berdasarkan nilai `val`.\n",
        "\n",
        "  Parameters:\n",
        "  - val (float): Nilai yang akan dievaluasi dalam PDF.\n",
        "  - lower_bound (float): Nilai batas bawah untuk distribusi uniform.\n",
        "  - upper_bound (float): Nilai batas atas untuk distribusi uniform.\n",
        "\n",
        "  Returns:\n",
        "  - pdf (float): Nilai probabilitas density function (PDF) untuk nilai dari variabel `val`. Mengembalikan 0 jika `val` di luar rentang [lower_bound, upper_bound]\n",
        "  \"\"\"\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "  if lower_bound <= val <= upper_bound:\n",
        "        pdf = 1 / (upper_bound - lower_bound)\n",
        "  else:\n",
        "        pdf = 0\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return pdf"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "metadata": {
        "id": "TZpASyt8AlX1"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "PDF untuk distribusi uniform dengan lower_bound=0 dan upper_bound=2 untuk nilai 0.1: 0.500\n",
            "PDF untuk distribusi uniform dengan lower_bound=15 dan upper_bound=30 untuk nilai 2: 0.000\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "print(f\"PDF untuk distribusi uniform dengan lower_bound={0} dan upper_bound={2} untuk nilai {1e-1}: {pdf_for_uniform(1e-1, 0, 2):.3f}\")\n",
        "print(f\"PDF untuk distribusi uniform dengan lower_bound={15} dan upper_bound={30} untuk nilai {2}: {pdf_for_uniform(2, 15, 30):.3f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "G62x1FEmGITA"
      },
      "source": [
        "#### Output yang diharapkan\n",
        "```\n",
        "PDF untuk distribusi uniform dengan lower_bound=0 dan upper_bound=2 untuk nilai 0.1: 0.500\n",
        "PDF untuk distribusi uniform dengan lower_bound=15 dan upper_bound=30 untuk nilai 2: 0.000\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0nkDrA3l_1Af"
      },
      "source": [
        "### Tugas 3: PDF untuk Distribusi Gaussian\n",
        "\n",
        "Jika sebuah variabel acak $X$ mengikuti distribusi Gaussian. Rumus untuk PDF-nya sebagai berikut.\n",
        "\n",
        "$$f(x;\\mu,\\sigma) = \\frac{1}{\\sigma \\sqrt{2 \\pi}} e^{-\\frac{1}{2}\\left(\\frac{x - \\mu}{\\sigma}\\right)^2}$$\n",
        "\n",
        "Ada tiga bagian utama dari rumus tersebut.\n",
        "- konstanta di depan: $\\frac{1}{\\sigma \\sqrt{2\\pi}}$\n",
        "- eksponensial: $e^{(\\cdots)}$\n",
        "- bagian dalam eksponen: $-\\tfrac{1}{2} \\big((x-\\mu)/\\sigma\\big)^2$\n",
        "\n",
        "Sama seperti sebelumnya, Anda perlu melengkapi fungsi di bawah ini dengan rumus yang sudah disebutkan."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "metadata": {
        "id": "cFlaBuATArR0"
      },
      "outputs": [],
      "source": [
        "def pdf_for_gaussian(val, miu, sigma):\n",
        "  \"\"\"\n",
        "  Menghitung probabily density function (PDF) untuk distribusi Gaussian berdasarkan nilai yang diberikan.\n",
        "\n",
        "  Parameters:\n",
        "  - val (float atau array-like): Nilai yang akan dievaluasi untuk PDF.\n",
        "  - miu (float): Nilai rata-rata untuk distribusi Gaussian.\n",
        "  - sigma (float): Nilai standar deviasi untuk distribusi Gaussian.\n",
        "\n",
        "  Returns:\n",
        "  - pdf (float atau array-like): Nilai PDF berdasarkan variabel `val` yang diberikan.\n",
        "  \"\"\"\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "  pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((val - miu) / sigma) ** 2)\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return pdf"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "id": "Jakja_P9Areb"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "PDF untuk distribusi Gaussian dengan miu=20 dan sigma=3 untuk nilai 10: 0.001\n",
            "PDF untuk distribusi Gaussian dengan miu=20 dan sigma=3 untuk nilai 0: 0.000\n",
            "PDF untuk distribusi Gaussian dengan miu=15 dan sigma=5 untuk nilai 1: 0.002\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "print(f\"PDF untuk distribusi Gaussian dengan miu={20} dan sigma={3} untuk nilai {10}: {pdf_for_gaussian(10, 20, 3):.3f}\")\n",
        "print(f\"PDF untuk distribusi Gaussian dengan miu={20} dan sigma={3} untuk nilai {0}: {pdf_for_gaussian(0, 20, 3):.3f}\")\n",
        "print(f\"PDF untuk distribusi Gaussian dengan miu={15} dan sigma={5} untuk nilai {1}: {pdf_for_gaussian(1, 15, 5):.3f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IWeadMHPIlih"
      },
      "source": [
        "#### Output yang diharapkan\n",
        "\n",
        "```\n",
        "PDF untuk distribusi Gaussian dengan miu=20 dan sigma=3 untuk nilai 10: 0.001\n",
        "PDF untuk distribusi Gaussian dengan miu=20 dan sigma=3 untuk nilai 0: 0.000\n",
        "PDF untuk distribusi Gaussian dengan miu=15 dan sigma=5 untuk nilai 1: 0.002\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-DGVy5YR_8n6"
      },
      "source": [
        "### Tugas 4: Probability Mass Function (PMF) untuk Distribusi Binomial\n",
        "\n",
        "Untuk distribusi binomial, karena ini adalah distribusi diskret, kita akan menggunakan Probability Mass Function (PMF) alih-alih PDF. Ingat bahwa jika sebuah variabel acak $X$ mengikuti distribusi binomial dengan parameter `n_trials (n)` dan `probability (p)`, rumus PMF yang didapat menjadi berikut.\n",
        "\n",
        "$$f(k; n, p) = {n \\choose k}  p^k  (1-p)^{n-k}$$\n",
        "\n",
        "Dengan ${n \\choose k} = \\frac{n!}{k!(n-k)!}$ dapat dihitung menggunakan beberapa fungsi dari Python, seperti berikut.\n",
        "\n",
        "*   math.factorial,\n",
        "*   scipy.special.comb, dan\n",
        "*   scipy.binom.pmf (Direkomendasikan)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "metadata": {
        "id": "X-xcZelOAs7b"
      },
      "outputs": [],
      "source": [
        "def pmf_for_binomial(val, n_trials, probability):\n",
        "  \"\"\"\n",
        "  Menghitung probability mass function (PMF) untuk distribusi binomial berdasarkan nilai pada variabel `val`\n",
        "\n",
        "  Parameters:\n",
        "  - val (int): Nilai yang akan dievaluasi untuk PMF.\n",
        "  - n_trials (int): Banyaknya percobaan dalam distribusi binomial.\n",
        "  - probability (float): Banyaknya percobaan yang berhasil.\n",
        "\n",
        "  Returns:\n",
        "  - pmf (float): Nilai probability mass function (PMF) untuk distribusi binomial untuk nilai pada variabel `val`.\n",
        "  \"\"\"\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "  if 0 <= val <= n_trials:\n",
        "        # Hitung kombinasi dengan rumus stabil numerik:\n",
        "        k = int(val)\n",
        "        n = int(n_trials)\n",
        "        comb = np.prod((np.arange(n - k + 1, n + 1)) / np.arange(1, k + 1))\n",
        "        \n",
        "        pmf = comb * (probability ** k) * ((1 - probability) ** (n - k))\n",
        "  else:\n",
        "        pmf = 0\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return pmf"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "id": "Afcx3ZvPAxO2"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "PMF untuk distribusi binomial dengan n_trials=15 dan probability=0.9 dengan nilai 15: 0.206\n",
            "PMF untuk distribusi binomial dengan n_trials=30 dan probability=0.5 dengan nilai 15: 0.144\n",
            "PMF untuk distribusi binomial dengan n_trials=15 dan probability=0.5 dengan nilai 20: 0.000\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "print(f\"PMF untuk distribusi binomial dengan n_trials={15} dan probability={0.9} dengan nilai {15}: {pmf_for_binomial(15, 15, 0.9):.3f}\")\n",
        "print(f\"PMF untuk distribusi binomial dengan n_trials={30} dan probability={0.5} dengan nilai {15}: {pmf_for_binomial(15, 30, 0.5):.3f}\")\n",
        "print(f\"PMF untuk distribusi binomial dengan n_trials={15} dan probability={0.5} dengan nilai {20}: {pmf_for_binomial(20, 15, 0.5):.3f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AYlKtSi-M8xh"
      },
      "source": [
        "#### Output yang diharapkan\n",
        "\n",
        "```\n",
        "PMF untuk distribusi binomial dengan n_trials=15 dan probability=0.9 dengan nilai 15: 0.206\n",
        "PMF untuk distribusi binomial dengan n_trials=30 dan probability=0.5 dengan nilai 15: 0.144\n",
        "PMF untuk distribusi binomial dengan n_trials=15 dan probability=0.5 dengan nilai 20: 0.000\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "E-htTlOCPqZX"
      },
      "source": [
        "### Tugas 5: Mengestimasikan Parameter Distribusi\n",
        "\n",
        "**Hebat!** Semua fungsi untuk menghasilkan PDF (dan PMF) dari masing-masing distribusi telah berhasil dibuat. Namun, untuk menggunakan setiap fungsi tersebut, kita perlu menyiapkan beberapa parameter, seperti\n",
        "\n",
        "- `miu` and `sigma` for the `wingspan_cm` feature\n",
        "- `miu` and `sigma` for the `weight_g` feature\n",
        "- `n_trials` and `probability` for the `sing_days` feature\n",
        "- `lower_bound` and `upper_bound` for the `beak_head_ratio` feature\n",
        "\n",
        "Oleh karena itu, tahap selanjutnya adalah mengestimasi parameter distribusi untuk digunakan pada fungsi-fungsi yang sudah Anda buat sebelumnya.\n",
        "\n",
        "> **Lho? Mengapa perlu mengestimasi parameter distribusi? Bukankah saat membuat dataset kita sudah tahu parameternya?**\n",
        "> Tenang! Ketika kita menerapkan algoritma Naive Bayes, kita menganggap **dataset sintetis** tersebut sebagai **dataset nyata** sehingga kita tidak mengetahui parameter distribusinya secara langsung. Meskipun kita sebenarnya tahu nilainya (karena kita yang menghasilkan datanya), proses ini merupakan latihan untuk meniru kondisi di dunia nyata, yaitu parameter distribusi harus diestimasi dari data."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jZBRQ-AOkNy0"
      },
      "source": [
        "Mari kita lihat contoh sederhana untuk memahami cara mengestimasi parameter dari masing-masing distribusi.\n",
        "\n",
        "Perhatikan penjelasan berikut mengenai rumus untuk menghitung setiap parameter.\n",
        "\n",
        "* **μ (miu)**: Nilai rata-rata dari sampel.\n",
        "* **σ (sigma)**: Nilai standar deviasi dari sampel.\n",
        "* **success\\_probability**: Probabilitas terjadinya sukses dalam distribusi binomial.\n",
        "* **lower\\_bound**: Nilai minimum dalam sampel.\n",
        "* **upper\\_bound**: Nilai maksimum dalam sampel.\n",
        "\n",
        "Sekarang, giliran Anda menghitung setiap parameter tersebut untuk fungsi-fungsi yang telah dibuat di bawah ini. Anda dapat memanfaatkan pustaka `numpy` untuk membantu melakukan perhitungan.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "metadata": {
        "id": "g1Ef0ow0Az2b"
      },
      "outputs": [],
      "source": [
        "def gaussian_params_estimation(sample):\n",
        "  \"\"\"\n",
        "  Mengestimasikan nilai miu (rata-rata) dan sigma (standar deviasi) untuk sampel yang diberikan.\n",
        "\n",
        "  Parameters:\n",
        "  - sample (ndarray): Array yang merepresentasikan data sampel.\n",
        "\n",
        "  Returns:\n",
        "  - miu (float): Nilai rata-rata sampel.\n",
        "  - sigma (float): Nilai standar deviasi sampel.\n",
        "  \"\"\"\n",
        "  # MULAI KODE DI SINI\n",
        "  miu = np.mean(sample)\n",
        "  sigma = np.std(sample, ddof=0)\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return miu, sigma\n",
        "\n",
        "\n",
        "def binomial_params_estimation(sample):\n",
        "  \"\"\"\n",
        "  Mengestimasikan parameter distribusi binomial dari sampel yang diberikan.\n",
        "\n",
        "  Parameter:\n",
        "  - sample (ndarray): Jumlah sampel percobaan.\n",
        "\n",
        "  Returns:\n",
        "  - n_trials (int): Jumlah percobaan dalam distribusi binomial (diasumsikan 30).\n",
        "  - success_probability (float): Estimasi probabilitas sukses per percobaan.\n",
        "  \"\"\"\n",
        "  n_trials = 30\n",
        "\n",
        "  # MULAI KODE DI SINI\n",
        "  success_probability = np.mean(sample) / n_trials\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return n_trials, success_probability\n",
        "\n",
        "\n",
        "def uniform_params_estimation(sample):\n",
        "  \"\"\"\n",
        "  Mengestimasikan parameter distribusi uniform dari sampel yang diberikan.\n",
        "\n",
        "  Parameter:\n",
        "  - sample (ndarray): Array berisi data sampel.\n",
        "\n",
        "  Returns:\n",
        "  - lower_bound (float): Nilai minimum dalam sampel.\n",
        "  - upper_bound (float): Nilai maksimum dalam sampel.\n",
        "  \"\"\"\n",
        "  # MULAI KODE DI SINI\n",
        "  lower_bound = np.min(sample)\n",
        "  upper_bound = np.max(sample)\n",
        "  # AKHIRI KODE DI SINI\n",
        "\n",
        "  return lower_bound, upper_bound\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "metadata": {
        "id": "qFOlxTaSAxcu"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Estimasi Gaussian:\n",
            "  Rata-rata (mu) = 47.400, Standar deviasi (sigma) = 1.810\n",
            "  Sampel yang digunakan = [45.2 47.8 50.1 46.5]\n",
            "\n",
            "Estimasi Binomial:\n",
            "  Jumlah percobaan (n) = 30, Probabilitas sukses (p) = 0.600\n",
            "  Sampel yang digunakan = [12 18 25 20 15]\n",
            "\n",
            "Estimasi Uniform:\n",
            "  Batas bawah (a) = 1.200, Batas atas (b) = 4.000\n",
            "  Sampel yang digunakan = [1.2 3.4 2.8 4.  3.1]\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "sample_gaussian = np.array([45.2, 47.8, 50.1, 46.5])\n",
        "miu, sigma = gaussian_params_estimation(sample_gaussian)\n",
        "print(f\"Estimasi Gaussian:\\n  Rata-rata (mu) = {miu:.3f}, Standar deviasi (sigma) = {sigma:.3f}\\n  Sampel yang digunakan = {sample_gaussian}\\n\")\n",
        "\n",
        "sample_binomial = np.array([12, 18, 25, 20, 15])\n",
        "n, p = binomial_params_estimation(sample_binomial)\n",
        "print(f\"Estimasi Binomial:\\n  Jumlah percobaan (n) = {n}, Probabilitas sukses (p) = {p:.3f}\\n  Sampel yang digunakan = {sample_binomial}\\n\")\n",
        "\n",
        "sample_uniform = np.array([1.2, 3.4, 2.8, 4.0, 3.1])\n",
        "a, b = uniform_params_estimation(sample_uniform)\n",
        "print(f\"Estimasi Uniform:\\n  Batas bawah (a) = {a:.3f}, Batas atas (b) = {b:.3f}\\n  Sampel yang digunakan = {sample_uniform}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KV_YaUeLpCtJ"
      },
      "source": [
        "#### Output yang diharapkan\n",
        "```\n",
        "Estimasi Gaussian:\n",
        "  Rata-rata (mu) = 47.400, Standar deviasi (sigma) = 1.810\n",
        "  Sampel yang digunakan = [45.2 47.8 50.1 46.5]\n",
        "\n",
        "Estimasi Binomial:\n",
        "  Jumlah percobaan (n) = 30, Probabilitas sukses (p) = 0.600\n",
        "  Sampel yang digunakan = [12 18 25 20 15]\n",
        "\n",
        "Estimasi Uniform:\n",
        "  Batas bawah (a) = 1.200, Batas atas (b) = 4.000\n",
        "  Sampel yang digunakan = [1.2 3.4 2.8 4.  3.1]\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9yxB2bLBP8Z4"
      },
      "source": [
        "### Tugas 6: Membuat Estimasi Parameter\n",
        "\n",
        "**Hebat!** Sekarang Anda sudah memiliki pemahaman tentang cara membuat estimasi parameter berdasarkan nilai-nilai fitur dalam data. Selanjutnya, mari kita implementasikan konsep yang sudah dipelajari pada cell sebelumnya dalam fungsi di bawah ini.\n",
        "\n",
        "Fungsi `training_params_estimation` bertujuan untuk **mengestimasi parameter distribusi** berdasarkan **data training** yang diberikan.\n",
        "\n",
        "Fungsi ini akan menerima dua parameter, yaitu\n",
        "\n",
        "* `dataframe`: DataFrame yang berisi data training lengkap.\n",
        "* `features`: daftar nama fitur yang ingin diproses dan dihitung parameternya.\n",
        "\n",
        "Dalam fungsi ini, kita akan melakukan **estimasi parameter** secara **terpisah untuk setiap fitur** dalam dataframe. Salah satu contohnya adalah fitur `sing_days`, fungsi akan menghitung **parameter distribusi binomial** karena sebelumnya kita mendefinisikan `sing_days` sebagai **variabel yang mengikuti distribusi binomial**.\n",
        "\n",
        "> Periksa variabel `breed_params` pada penjelesan segmen 2.1\n",
        "\n",
        "---\n",
        "\n",
        "Selain itu, fungsi ini akan mengembalikan dua dictionary berupa `params_dict` dan `probs_dict`.\n",
        "\n",
        "* `params_dict` akan menyimpan hasil estimasi parameter untuk setiap fitur pada tiap jenis burung (breed). Variabel ini akan memiliki nested dictionary dengan keterangan lebih lanjut adalah berikut.\n",
        "* Level pertama berisi *key* berupa label jenis (direpresentasikan sebagai numerik/integer).\n",
        "* Nilai pada level pertama adalah dictionary lain yang berisi nama-nama fitur.\n",
        "* Untuk setiap fitur, nilainya adalah objek dataclass yang berisi parameter-parameter yang sudah diestimasi (misalnya rata-rata dan standar deviasi untuk distribusi Gaussian).\n",
        "```\n",
        "{\n",
        "  0: {\n",
        "    'weight_g': params_dataclass(param1=x41, param2=x42)\n",
        "    'sing_days': params_dataclass(param1=x11, param2=x12),\n",
        "    'beak_head_ratio': params_dataclass(param1=x21, param2=x22),\n",
        "    'wingspan_cm': params_dataclass(param1=x31, param2=x32),\n",
        "  },\n",
        "  1: ...\n",
        "}\n",
        "```\n",
        "\n",
        "* `probs_dict` menyimpan informasi tentang proporsi data untuk tiap jenis burung.\n",
        "* Dictionary ini memetakkan setiap label ras pada nilai proporsinya (direpresentasikan dalam angka desimal antara 0 dan 1).\n",
        "* Total semua nilai dalam probs_dict seharusnya mendekati atau sama dengan 1, karena mewakili distribusi probabilitas semua kelas.\n",
        "\n",
        "$$\n",
        "{\n",
        "  0: 0.25,\n",
        "  1: 0.5,\n",
        "  2: 0.25\n",
        "}\n",
        "$$\n",
        "\n",
        "Untuk menyelesaikan fungsi ini, perhatikan setiap komentar yang berisi petunjuk atau langkah-langkah yang harus dilakukan. Anda bisa menggunakan berbagai pendekatan yang sesuai, asalkan tujuan akhirnya tercapai. Sebagai contoh, Anda bisa menggunakan `match` statement untuk membuat logika seperti switch-case atau memilih menggunakan struktur `if-else` jika lebih nyaman."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "id": "cfTCAXn3J-B9"
      },
      "outputs": [],
      "source": [
        "def training_params_estimation(df, features):\n",
        "    \"\"\"\n",
        "    Menghitung parameter estimasi untuk melatih model berdasarkan dataframe dan fitur yang diberikan.\n",
        "\n",
        "    Parameters:\n",
        "    - df (pandas.DataFrame): Dataframe yang berisi data pelatihan.\n",
        "    - features (list): Daftar nama fitur yang akan dipertimbangkan.\n",
        "\n",
        "    Returns:\n",
        "        tuple: Sebuah tuple yang berisi dua dictionary:\n",
        "            - params_dict (dict): Dictionary yang berisi parameter estimasi untuk setiap ras dan fitur.\n",
        "            - probs_dict (dict): Dictionary yang berisi proporsi data yang termasuk dalam setiap ras.\n",
        "    \"\"\"\n",
        "\n",
        "    # Dictionary yang akan menyimpan parameter estimasi\n",
        "    params_dict = {}\n",
        "\n",
        "    # Dictionary yang akan menyimpan proporsi data untuk setiap kelas\n",
        "    probs_dict = {}\n",
        "\n",
        "    # MULAI KODE DI SINI\n",
        "\n",
        "    # Lakukan loop/iterasi pada setiap nilai di kolom \"breed\" dari dataframe.\n",
        "\n",
        "\n",
        "        # Filter dataframe berdasarkan ras (breed) dan kolom fitur tertentu.\n",
        "        # Misalnya: df_dog_breed = df[df[\"breed\"] == \"Bulldog\"][features]\n",
        "        # Untuk referensi slicing* dengan pandas, bisa menggunakan fungsi df_breed.groupby diikuti .get_group\n",
        "        # atau menggunakan sintaks df[df['breed'] == group]\n",
        "\n",
        "\n",
        "        # Simpan probabilitas (proporsi) setiap kelas (ras) dalam dictionary probs_dict yang sudah didefinisikan di atas.\n",
        "        # Proporsi dihitung dengan: jumlah baris ras ini ÷ jumlah baris seluruh dataframe.\n",
        "        # Jumlah baris dataframe bisa diperoleh dengan fungsi len().\n",
        "        # Contoh: probs_dict[0] = 20/100 = 0.2\n",
        "\n",
        "\n",
        "        # Inisialisasi dictionary bagian dalam\n",
        "    inner_dict = {} #JANGAN DIHAPUS\n",
        "\n",
        "        # Lakukan loop untuk setiap kolom pada dataframe yang sudah di-slice\n",
        "        # Kolom-kolom dataframe bisa didapat dengan dataframe.columns\n",
        "        # Contoh: for feature in df_dog_breed.columns:\n",
        "\n",
        "            # Percabangan untuk setiap fitur\n",
        "\n",
        "                    # Untuk fitur \"wingspan_cm\" dan \"weight_g\" yang mengikuti distribusi Gaussian:\n",
        "                    # - Hitung nilai rata-rata (μ) dan standar deviasi (σ) dari kolom fitur tersebut.\n",
        "                    # - Gunakan nilai μ dan σ untuk membuat objek parameter dengan class gaussian_params.\n",
        "                    #\n",
        "                    # Contoh penerapan:\n",
        "                    # mu = df_breed[feature].mean()   # rata-rata dari kolom fitur\n",
        "                    # sigma = df_breed[feature].std()  # standar deviasi dari kolom fitur. Pastikan gunakan standar deviasi sampel.\n",
        "                    # params = gaussian_params(miu=mu, sigma=sigma)\n",
        "                    #\n",
        "                    # Catatan: gunakan df_breed agar bisa lebih spesifik dengan output yang diharapkan.\n",
        "\n",
        "\n",
        "                    # Untuk fitur \"sing_days\" yang mengikuti distribusi binomial.\n",
        "                    # - Tentukan nilai n_trials (jumlah percobaan), biasanya bisa diambil dari nilai maksimum kolom ini.\n",
        "                    # - Hitung peluang p, misalnya dengan membagi nilai rata-rata kolom dengan n.\n",
        "                    # - Gunakan nilai n_trials dan probability ke dalam objek binomial_params.\n",
        "\n",
        "\n",
        "                    # Untuk fitur \"beak_head_ratio\" yang mengikuti distribusi uniform.\n",
        "                    # - Tentukan nilai batas bawah (lower_bound) dengan nilai minimum dari kolom.\n",
        "                    # - Tentukan nilai batas atas (upper_bound) dengan nilai maksimum dari kolom.\n",
        "                    # - Gunakan nilai batas atas dan batas bawah ke dalam objek uniform_params.\n",
        "\n",
        "    for breed in df['breed'].unique():\n",
        "        # Filter dataframe berdasarkan ras\n",
        "        df_breed = df[df[\"breed\"] == breed][features]\n",
        "\n",
        "        # Hitung proporsi (probabilitas) ras ini terhadap total data\n",
        "        probs_dict[breed] = len(df_breed) / len(df)\n",
        "\n",
        "        # Inisialisasi dictionary bagian dalam\n",
        "        inner_dict = {}\n",
        "\n",
        "        # Loop setiap fitur\n",
        "        for feature in features:\n",
        "            # Fitur dengan distribusi Gaussian\n",
        "            if feature in [\"wingspan_cm\", \"weight_g\"]:\n",
        "                mu = df_breed[feature].mean()\n",
        "                sigma = df_breed[feature].std(ddof=1)\n",
        "                params = gaussian_params(miu=mu, sigma=sigma)\n",
        "\n",
        "            # Fitur dengan distribusi Binomial\n",
        "            elif feature == \"sing_days\":\n",
        "                n_trials = df_breed[feature].max()\n",
        "                probability = df_breed[feature].mean() / n_trials\n",
        "                params = binomial_params(n_trials=n_trials, probability=probability)\n",
        "\n",
        "            # Fitur dengan distribusi Uniform\n",
        "            elif feature == \"beak_head_ratio\":\n",
        "                lower_bound = df_breed[feature].min()\n",
        "                upper_bound = df_breed[feature].max()\n",
        "                params = uniform_params(lower_bound=lower_bound, upper_bound=upper_bound)\n",
        "\n",
        "            # Simpan objek dataclass dalam inner_dict\n",
        "            inner_dict[feature] = params\n",
        "\n",
        "        # Simpan inner_dict dalam params_dict untuk ras saat ini\n",
        "        params_dict[breed] = inner_dict\n",
        "\n",
        "    ### AKHIRI KODE DI SINI ###\n",
        "\n",
        "    return params_dict, probs_dict"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 34,
      "metadata": {
        "collapsed": true,
        "id": "1e_GSU8UA5kj"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Parameter distribusi:\n",
            "\n",
            "{np.int64(0): {'beak_head_ratio': uniform_params(lower_bound=0.101, upper_bound=0.599),\n",
            "               'sing_days': binomial_params(n_trials=29.000, probability=0.828),\n",
            "               'weight_g': gaussian_params(mu=19.998, sigma=1.019),\n",
            "               'wingspan_cm': gaussian_params(mu=34.997, sigma=1.528)},\n",
            " np.int64(1): {'beak_head_ratio': uniform_params(lower_bound=0.200, upper_bound=0.500),\n",
            "               'sing_days': binomial_params(n_trials=23.000, probability=0.651),\n",
            "               'weight_g': gaussian_params(mu=24.956, sigma=4.959),\n",
            "               'wingspan_cm': gaussian_params(mu=29.982, sigma=1.984)},\n",
            " np.int64(2): {'beak_head_ratio': uniform_params(lower_bound=0.100, upper_bound=0.300),\n",
            "               'sing_days': binomial_params(n_trials=17.000, probability=0.516),\n",
            "               'weight_g': gaussian_params(mu=31.699, sigma=3.118),\n",
            "               'wingspan_cm': gaussian_params(mu=39.649, sigma=3.638)}}\n",
            "\n",
            "Probabilitas untuk setiap kelas:\n",
            "\n",
            "{np.int64(0): 0.34575569358178054,\n",
            " np.int64(1): 0.39544513457556935,\n",
            " np.int64(2): 0.2587991718426501}\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "import pprint\n",
        "pp = pprint.PrettyPrinter()\n",
        "\n",
        "train_params, train_class_probs = training_params_estimation(df_train, FEATURES)\n",
        "\n",
        "print(\"Parameter distribusi:\\n\")\n",
        "pp.pprint(train_params)\n",
        "print(\"\\nProbabilitas untuk setiap kelas:\\n\")\n",
        "pp.pprint(train_class_probs)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HBXnvAz1SUsZ"
      },
      "source": [
        "```\n",
        "Parameter distribusi:\n",
        "\n",
        "{0: {'beak_head_ratio': uniform_params(lower_bound=0.101, upper_bound=0.599),\n",
        "     'sing_days': binomial_params(n_trials=29.000, probability=0.800),\n",
        "     'weight_g': gaussian_params(mu=19.998, sigma=1.019),\n",
        "     'wingspan_cm': gaussian_params(mu=34.997, sigma=1.528)},\n",
        " 1: {'beak_head_ratio': uniform_params(lower_bound=0.200, upper_bound=0.500),\n",
        "     'sing_days': binomial_params(n_trials=23.000, probability=0.499),\n",
        "     'weight_g': gaussian_params(mu=24.956, sigma=4.959),\n",
        "     'wingspan_cm': gaussian_params(mu=29.982, sigma=1.984)},\n",
        " 2: {'beak_head_ratio': uniform_params(lower_bound=0.100, upper_bound=0.300),\n",
        "     'sing_days': binomial_params(n_trials=17.000, probability=0.292),\n",
        "     'weight_g': gaussian_params(mu=31.699, sigma=3.118),\n",
        "     'wingspan_cm': gaussian_params(mu=39.649, sigma=3.638)}}\n",
        "\n",
        "Probabilitas untuk setiap kelas:\n",
        "\n",
        "{0: 0.34575569358178054, 1: 0.39544513457556935, 2: 0.2587991718426501}\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yeupaQFhcRr_"
      },
      "source": [
        "### Tugas 7 - Posterior Probability $P(X \\mid C_i)$:  Menghitung Probabilitas Jenis Burung berdasarkan Atributnya\n",
        "\n",
        "Untuk mengimplementasikan **Naive Bayes Classifier**, kita menggunakan asumsi **independensi bersyarat pada kelas** (*class-conditional independence*). Artinya, untuk suatu pengamatan $x = (x_1, \\ldots, x_n)$ pada $X$, kita bisa menghitung peluang $x$ diberikan kelas tertentu ($C_i$) dengan cara berikut.\n",
        "\n",
        "$$\n",
        "P(x \\mid C_{i}) = P(x_1 \\mid C_i) \\cdot P(x_2 \\mid C_i) \\cdot \\ldots \\cdot P(x_n \\mid C_i) = \\prod_{k = 1}^{n} P(x_k \\mid C_i)\n",
        "$$\n",
        "\n",
        "Probabilitas masing-masing atribut $P(x_k \\mid C_i)$ dapat diestimasi dari data latih (training data).\n",
        "\n",
        "Jika $x_k$ bernilai kontinu atau diskret, kita perlu membuat asumsi tentang bentuk distribusinya dan menghitung parameternya dari data latih. Misalnya, jika $x_k$ bersifat kontinu, biasanya diasumsikan bahwa $P(x_k \\mid C_i)$ mengikuti distribusi Gaussian dengan parameter rata-rata $\\mu_{C_i}$ dan standar deviasi $\\sigma_{C_i}$.\n",
        "\n",
        "Oleh karena itu, kita perlu memperkirakan nilai $\\mu$ dan $\\sigma$ dari data latih, kemudian menghitung dengan rumus berikut.\n",
        "\n",
        "$$\n",
        "P(x_k \\mid C_i) = \\text{PDF}_{\\text{gaussian}}(x_k,\\mu_{C_i},\\sigma_{C_i}).\n",
        "$$\n",
        "\n",
        "Dalam konteks latihan ini, kita sudah mengetahui bentuk distribusi untuk setiap fitur. Dalam kata lain, tugas kita adalah menghitung nilai `PDF` yang sesuai untuk masing-masing fitur dengan menggunakan parameter hasil estimasi yang telah tersedia sebelumnya. Parameter-parameter ini tinggal dimasukkan pada fungsi PDF yang sesuai untuk mendapatkan nilainya.\n",
        "\n",
        "Silakan lengkapi fungsi `probability_given_class` di bawah. Fungsi ini membutuhkan beberapa parameter berikut.\n",
        "\n",
        "* `X`: sebuah list yang berisi nilai-nilai untuk setiap fitur (dengan urutan sesuai `features`)\n",
        "* `features`: daftar nama fitur yang akan diproses\n",
        "* `breed`: jenis burung (kelas) yang ingin kita hitung probabilitasnya\n",
        "* `params_dict`: dictionary yang memuat parameter hasil estimasi dari data training\n",
        "\n",
        "Fungsi ini harus mengembalikan nilai probabilitas (likelihood) untuk pengamatan `X` **dengan asumsi** bahwa ia berasal dari kelas (jenis burung) tertentu.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 35,
      "metadata": {
        "id": "oUupHObfgEZe"
      },
      "outputs": [],
      "source": [
        "def probability_given_class(X, features, breed, params_dict):\n",
        "    \"\"\"\n",
        "    Menghitung probabilitas bersyarat dari X berdasarkan jenis burung (breed),\n",
        "    menggunakan fitur dan parameter yang diberikan.\n",
        "\n",
        "    Parameters:\n",
        "    - X (list): List nilai-nilai fitur yang ingin dihitung probabilitasnya.\n",
        "    - features (list): List nama fitur yang sesuai dengan nilai-nilai pada X.\n",
        "    - breed (str): Jenis burung (kelas) yang menjadi acuan perhitungan probabilitas.\n",
        "    - params_dict (dict): Dictionary yang berisi parameter estimasi untuk setiap breed dan fitur.\n",
        "\n",
        "    Returns:\n",
        "    - Probability (float): Nilai probabilitas bersyarat X jika diketahui berasal dari jenis burung tersebut.\n",
        "    \"\"\"\n",
        "\n",
        "    if len(X) != len(features):\n",
        "        print(\"X dan daftar fitur harus memiliki panjang yang sama\")\n",
        "        return 0\n",
        "\n",
        "    # Inisialisasi probabilitas total\n",
        "    probability = 1.0\n",
        "\n",
        "    # MULAI KODE DI SINI\n",
        "    # Lakukan perulangan untuk setiap X dan features.\n",
        "    for x, feature in zip(X, features):\n",
        "        # Ambil breed dan freature yang sesuai dari params_dict yang diberikan.\n",
        "        # Contoh: params = params_dict[\"Bulldog\"][\"Bark\"]\n",
        "        params = params_dict[breed][feature]\n",
        "        # Percabangan untuk setiap fitur.\n",
        "        if hasattr(params, \"miu\") and hasattr(params, \"sigma\"):\n",
        "                # Hitung pdf sesuai distribusi gaussian menggunakan parameter estimasi dari tahapan sebelumnya.\n",
        "                # Contoh: probability_f = pdf_for_gaussian(x, params.miu, params.sigma)\n",
        "                probability_f = pdf_for_gaussian(x, params.miu, params.sigma)\n",
        "        elif hasattr(params, \"n\") and hasattr(params, \"p\"):   \n",
        "                # Hitung pmf untuk distribusi binomial menggunakan parameter estimasi dari tahapan sebelumnya.\n",
        "                probability_f = pmf_for_binomial(x, params.n, params.p)\n",
        "        elif hasattr(params, \"a\") and hasattr(params, \"b\"):\n",
        "                # Hitung pdf untuk distribusi uniform dengan parameter estimasi dari tahapan sebelumnya.\n",
        "                probability_f = pdf_for_uniform(x, params.a, params.b)\n",
        "\n",
        "\n",
        "        # Kalikan hasil pdf/pmf fitur ini ke total probabilitas\n",
        "        probability *= probability_f\n",
        "\n",
        "    # AKHIRI KODE DI SINI\n",
        "\n",
        "    return probability"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 36,
      "metadata": {
        "id": "8BIgUAVqCpFp"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Contoh burung memiliki jenis 2 dengan fitur: wingspan_cm = 46.98, weight_g = 37.98, sing_days = 14.00, beak_head_ratio = 0.30\n",
            "\n",
            "Probabilitas fitur-fitur ini jika burung diklasifikasikan sebagai jenis 0: 5.076647405144422e-219\n",
            "Probabilitas fitur-fitur ini jika burung diklasifikasikan sebagai jenis 1: 3.7710008983060214e-25\n",
            "Probabilitas fitur-fitur ini jika burung diklasifikasikan sebagai jenis 2: 6.783791481513813e-08\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "example_bird = df_test[FEATURES].loc[0]\n",
        "breed_bird = df_test[[\"breed\"]].loc[0][\"breed\"]\n",
        "\n",
        "print(f\"Contoh burung memiliki jenis {breed_bird} dengan fitur: wingspan_cm = {example_bird['wingspan_cm']:.2f}, weight_g = {example_bird['weight_g']:.2f}, sing_days = {example_bird['sing_days']:.2f}, beak_head_ratio = {example_bird['beak_head_ratio']:.2f}\\n\")\n",
        "\n",
        "print(f\"Probabilitas fitur-fitur ini jika burung diklasifikasikan sebagai jenis 0: {probability_given_class([*example_bird], FEATURES, 0, train_params)}\")\n",
        "print(f\"Probabilitas fitur-fitur ini jika burung diklasifikasikan sebagai jenis 1: {probability_given_class([*example_bird], FEATURES, 1, train_params)}\")\n",
        "print(f\"Probabilitas fitur-fitur ini jika burung diklasifikasikan sebagai jenis 2: {probability_given_class([*example_bird], FEATURES, 2, train_params)}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "KLzjhznDSb26"
      },
      "source": [
        "```\n",
        "Contoh burung memiliki jenis 2 dengan fitur: wingspan_cm = 46.98, weight_g = 37.98, sing_days = 14.00, beak_head_ratio = 0.30\n",
        "\n",
        "Probabilitas fitur-fitur ini jika burung diklasifikasikan sebagai jenis 0: 1.9252029362842843e-86\n",
        "Probabilitas fitur-fitur ini jika burung diklasifikasikan sebagai jenis 1: 1.863471980693772e-20\n",
        "Probabilitas fitur-fitur ini jika burung diklasifikasikan sebagai jenis 2: 9.672973185945612e-09\n",
        "```"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8UIlTMibBapa"
      },
      "source": [
        "#### Tugas 8: Prediksi Jenis Burung\n",
        "\n",
        "Hebat! Naive Bayes sudah diimplementasikan dalam sebuah fungsi dengan baik!\n",
        "\n",
        "Jika semua kelas memiliki jumlah data yang benar-benar seimbang, fungsi sebelumnya bisa digunakan langsung untuk menghitung *posterior* maksimum. Namun, **itu bukanlah kondisi yang kita hadapi di sini** sehingga kita masih perlu mengalikan setiap probabilitas \\$P(x \\mid C\\_{i})\\$ dengan probabilitas awal (prior) dari masing-masing kelas \\$P(C\\_{i})\\$.\n",
        "\n",
        "Pada akhirnya, rumus yang perlu kita maksimalkan untuk mendapatkan prediksi adalah berikut.\n",
        "\n",
        "$$\n",
        "P(x \\mid C_{i}) \\times P(C_{i})\n",
        "$$\n",
        "\n",
        "Kita dapat melakukan ini melalui cara mengalikan hasil dari `probability_given_class` dengan proporsi kelas sesuai dengan yang sudah disimpan dalam dictionary `probs_dict`.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "#### Tugas 8: Prediksi Jenis Burung\n",
        "\n",
        "Hebat! Naive Bayes sudah diimplementasikan dalam sebuah fungsi dengan baik!\n",
        "\n",
        "Jika semua kelas memiliki jumlah data yang benar-benar seimbang, fungsi sebelumnya bisa digunakan langsung untuk menghitung *posterior* maksimum. Namun, **itu bukanlah kondisi yang kita hadapi di sini** sehingga kita masih perlu mengalikan setiap probabilitas \\$P(x \\mid C\\_{i})\\$ dengan probabilitas awal (prior) dari masing-masing kelas \\$P(C\\_{i})\\$.\n",
        "\n",
        "Pada akhirnya, rumus yang perlu kita maksimalkan untuk mendapatkan prediksi adalah berikut.\n",
        "\n",
        "$$\n",
        "P(x \\mid C_{i}) \\times P(C_{i})\n",
        "$$\n",
        "\n",
        "Kita dapat melakukan ini melalui cara mengalikan hasil dari `probability_given_class` dengan proporsi kelas sesuai dengan yang sudah disimpan dalam dictionary `probs_dict`.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 37,
      "metadata": {
        "id": "TGwa7EzEJ-Ob"
      },
      "outputs": [],
      "source": [
        "def breed_bird_prediction(X, features, params_dict, probs_dict):\n",
        "    \"\"\"\n",
        "    Memprediksi jenis burung berdasarkan data input dan fitur-fitur yang diberikan.\n",
        "\n",
        "    Parameters:\n",
        "    - X (list atau array): Data input (nilai-nilai fitur) yang ingin diprediksi.\n",
        "    - features (list atau array): Daftar nama fitur yang digunakan untuk prediksi.\n",
        "    - params_dict (dict): Dictionary yang berisi parameter distribusi untuk setiap jenis burung.\n",
        "    - probs_dict (dict): Dictionary yang berisi peluang (proporsi) masing-masing jenis burung dalam data latih.\n",
        "\n",
        "    Returns:\n",
        "    - prediction (int): Indeks jenis burung yang diprediksi (contoh: 0, 1, atau 2).\n",
        "    \"\"\"\n",
        "\n",
        "    # MULAI KODE DI SINI\n",
        "\n",
        "    # Hitung nilai posterior untuk setiap jenis burung (0,1, dan 2)\n",
        "    # Petunjuk: Gunakan fungsi probability sebelumnya lalu kalikan hasilnya dengan proporsi data untuk setiap jenis burung (probs_dict).\n",
        "    posterior_breed_0 = probability_given_class(X, features, 0, params_dict) * probs_dict[0]\n",
        "    posterior_breed_1 = probability_given_class(X, features, 1, params_dict) * probs_dict[1]\n",
        "    posterior_breed_2 = probability_given_class(X, features, 2, params_dict) * probs_dict[2]\n",
        "\n",
        "    # Simpan semua nilai posterior ke dalam numpy array\n",
        "    # Kemudian ambil indeks dengan nilai tertinggi sebagai hasil prediksi\n",
        "    posterior = np.array([posterior_breed_0, posterior_breed_1, posterior_breed_2])\n",
        "    prediction = np.argmax(posterior)\n",
        "\n",
        "\n",
        "    # AKHIRI KODE DI SINI\n",
        "\n",
        "    return prediction\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 38,
      "metadata": {
        "id": "4pQUWoZcDvkO"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Contoh burung memiliki jenis 2 dan Naive Bayes mengklasifikasikannya sebagai 2\n"
          ]
        }
      ],
      "source": [
        "# Pengujian: Silakan gunakan cell ini untuk menguji fungsi yang Anda buat. Pastikan output-nya sesuai dengan yang diharapkan.\n",
        "\n",
        "example_pred = breed_bird_prediction([*example_bird], FEATURES, train_params, train_class_probs)\n",
        "print(f\"Contoh burung memiliki jenis {breed_bird} dan Naive Bayes mengklasifikasikannya sebagai {example_pred}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bOQGrqmSO0MT"
      },
      "source": [
        "# Evaluasi\n",
        "\n",
        "Anda telah menyelesaikan seluruh tahapan implementasi Naive Bayes. Algoritma tersebut sudah berhasil dijalankan pada satu data. Lalu, bagaimana hasilnya jika kita gunakan pada satu data testing? Mari jalankan kode berikut untuk melihatnya."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 40,
      "metadata": {
        "id": "1a4ZJURPDxX6"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Accuracy score for the test split: 0.91\n"
          ]
        }
      ],
      "source": [
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "preds = df_test.apply(lambda x: breed_bird_prediction([*x[FEATURES]], FEATURES, train_params, train_class_probs), axis=1)\n",
        "test_acc = accuracy_score(df_test[\"breed\"], preds)\n",
        "print(f\"Accuracy score for the test split: {test_acc:.2f}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "olWySR_KPYR1"
      },
      "source": [
        "Selamat! Anda berhasil menerapkan algoritma Naive Bayes dengan pendekatan yang sangat matematis dan mendapatkan akurasi hampir (atau bahkan mencapai) 100%.\n",
        "\n",
        "Anda mungkin merasa ragu karena hasil akurasi yang sempurna tampak tidak realistis. Hal ini sebenarnya wajar karena kita menggunakan data sintetis yang dibuat sendiri. Data dunia nyata tentu jauh lebih kompleks. Tetap semangat!"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "bmds",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.9.25"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
