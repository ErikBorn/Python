from PyPDF2 import PdfReader, PdfWriter

filename = "hstranscripts.pdf"
sig = "Q1 Transcript"
pageCount = 1
inputpdf = open(filename, "rb")
pagesCount = PdfReader(open(filename, "rb"))

L = ["Angelier, Natalie","Babbitt, Olivia","Bates, Elena","Beacom, Mary","Billig, Teagan","Bollin, McKenna","Boyd, Lucille","Boyer, Narelle","Brady, Addison","Callender, Kate","Cooper, Chanel","Corrales, Larissa","Crysel, Madison","Damon, Annika","Esteve, Ana","Frasier, Katie","Gieser, Madeline","Hagadorn, Stella","Hamrick, Abigail","Haro, Malika","Harp, Vivian","Harrington, Alexandra","Harry, Millicent","High, Madison","Humphreys, Lauren","James, Jenica","Kennedy, JoAnna","Kirschbaum, Ella","Lowe, Graeson","Martin, Aubrey","Martinez, Emalisa","McCrea, Chloe","Mechem, Theodora","Millradt, Lorna","Novak, Lucille","Osborn, Mary","Pallotta, Madeleine","Parrott, Amelia","Perry, Mia","Phoenix, India","Pilarowski, Lydia","Porter, Emma","Quiroz, Camila","Ream, Catherine","Rechtin, Frances","Reed, Lila","Riehl, Selah","Ritchie, Sophia","Rivas Quijada, Dafne","Saucedo, Gabriella","Sheppard, Lillie","Sims, Gloria","Smith, Sidney","Spiel, Mila","Stein-Plog, Gabrielle","Swanson, Ella","Syre, Elle","Toler, Greta","Turner, Violet","Ursua Garcia, Maya","Valdez, Neve'","Viehl, Corinne","Walton, Zoe","Williams, Mia","Wilten, Claire"]

counter = 0
for i in range(0,len(pagesCount.pages),pageCount,):
    output = PdfWriter()
    output.append(fileobj=inputpdf, pages=(i, i+pageCount))
    with open(L[counter]+" "+ sig +".pdf", "wb") as outputStream:
        output.write(outputStream)
    counter+=1

