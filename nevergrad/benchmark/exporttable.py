# Utils for exporting a data table.
# Used in competence maps.

def remove_parens(data):
    try:
        data = [[d[:d.index("(")] for d in datarow] for datarow in data]
    except:
        pass
    return data
    

def export_table(filename, rows, cols, data):
    """Exports data in filename with rows and cols as described.
    More precisely, rows specifies the row names, cols specifies the col names,
    and data[i][j] corresponds to the data in row rows[i] and col cols[j].
    """
    rows = [str(r) for r in rows]
    cols = [str(r) for r in cols]
    data = [[d.replace("%", "\%").replace("_", "") for d in datarow] for datarow in data]
    data = remove_parens(data)
    print("filename=", filename)
    print("rows=", rows)
    print("cols=", cols)
    print("data=", data)
    with open(filename, "w") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{lscape}\n")
        f.write("\\usepackage{array}\n")
        f.write("\\lccode`0=`0\n")
        f.write("\\lccode`1=`1\n")
        f.write("\\lccode`2=`2\n")
        f.write("\\lccode`3=`3\n")
        f.write("\\lccode`4=`4\n")
        f.write("\\lccode`5=`5\n")
        f.write("\\lccode`6=`6\n")
        f.write("\\lccode`7=`7\n")
        f.write("\\lccode`8=`8\n")
        f.write("\\lccode`9=`9\n")
        f.write("\\newcolumntype{P}[1]{>{\hspace{0pt}}p{#1}}\n")
        f.write("\\begin{document}\n")
        f.write("\\scriptsize\n")
        f.write("\\renewcommand{\\arraystretch}{1.5}\n")
        f.write("\\sloppy\n")
        p = str(1./(2+len(cols)))
        # f.write("\\begin{landscape}\n")
        f.write("\\begin{tabular}{|P{" + p +"\\textwidth}|" + ("P{" + p + "\\textwidth}|") * len(cols) + "}\n")
        f.write("\\hline\n")
        f.write(" & " + "&".join(cols) + "\\\\\n")
        f.write("\\hline\n")
        for i, row in enumerate(rows):
            f.write(row + "&" + "&".join([re.sub("[A-Z]", lambda w: " " + w.group(), d) for d in data[i]]) + "\\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        # f.write("\\end{landscape}\n")
        f.write("\\end{document}\n")

