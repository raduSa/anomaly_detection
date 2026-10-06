import fitz

src = r"C:\Users\Radu Savin\facultate\anomaly_detection\thesis\shortened_final_version\final_version.pdf"
outdir = r"C:\Users\Radu Savin\facultate\anomaly_detection\thesis\other_docs\assets"
doc = fitz.open(src)

ZOOM = 4.0
mat = fitz.Matrix(ZOOM, ZOOM)

def crop(pageno, bbox, fname, pad=4):
    page = doc[pageno - 1]
    rect = fitz.Rect(bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    pix.save(f"{outdir}\\{fname}")
    print("saved", fname, pix.width, pix.height)

page24 = doc[23]
print(24, [(i['bbox'], i.get('xref')) for i in page24.get_image_info(xrefs=True)])

# Figure 5.1 - Gini histogram, page 21
crop(21, (188.1, 278.8, 407.2, 388.4), "gini_hist.png")

# Figure 5.2 - Temporal IForest SHAP twin line charts, page 23
crop(23, (111.9, 59.5, 483.4, 158.7), "shap_temporal_iforest.png")

# Figure 7.1 - Yahoo heatmap, page 37 (two panels)
crop(37, (239.9, 268.4, 355.4, 557.6), "yahoo_heatmap.png")
