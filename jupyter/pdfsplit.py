from PyPDF2 import PdfReader, PdfWriter

# reader = PdfReader("out1.pdf")
# writer = PdfWriter()

# for page in reader.pages:
#     page.compress_content_streams()  # This is CPU intensive!
#     writer.add_page(page)

# with open("out1b.pdf", "wb") as f:
#     writer.write(f)

filename = "madys.pdf"
sig = "S1 AY22-23"
pageCount = 3
inputpdf = open(filename, "rb")
pagesCount = PdfReader(open(filename, "rb"))

# L = ["Adam, Katrine", "Alvariza, Giselle Karine", "Anandan, Isha", "Bennett-Spurlock, Saffron", "Bidgood, Ainsley", "Bonilla Ortega, Emily", "Brennan, Ainsley", "Burke, Sara Ashley", "Callender, Chloe", "Conrad, Naomi Mei Jian", "Daly, Sophia", "Davis, Kailee", "Farkash, Kiana Yasmin", "Forschner, Anneliese", "Garcia, Anahi", "Garofalo, Lauren", "Gesler, Margaret", "Gonzalez Zamora, Fatima", "Goodreau, Olivia", "Gunter, Kaitlyn Grace", "Hallam, Molly Jane", "Hay, Katherine", "Hodlewsky, Ariana", "Husmann, Hailey", "Husmann, Hana", "Jung, Greta", "Ketterhagen, Taylor Siena", "Kiefer, Lucy", "Kim, Bridget", "Lawton, Ava Joan", "Madden, Molly", "Maestas, Sophia", "Malik, Claire Elizabeth", "Marshall, Jordan", "Martin, Aenor", "McClure, Iris", "Mills, Gracen Madeleine", "Montross, Nora", "Parsons, Isabel Jane", "Perry, Jewelie Robin", "Pizarro, Belen Sofia", "Price, Addison L", "Rauzzino, Perry Annalee", "Robles, Emily Isabela", "Sanchez, Jaden", "Santiago, Ana", "Saucedo, Ana Isabella", "Schuett, Hannah", "Sharma, Paree", "Smith, Emerson", "Stander, Sara", "Stefanoudakis, Maria Ioanna", "Ursua Garcia, Andrea Yaretzi", "Velmurugan, Ilana", "Walker, Lucille", "Walling, Caitlin", "Williams, Jaelyn", "Xie, Victoria", "Yearsley, Ella Marlo"]
#L = ["Aldrich, Drew Gabriel", "Anderson, Dahlia Louise Collins", "Arney, Kyler Kay", "Buttitta, Oliver Rolfs", "Carrell, Maeve Margaret", "Hamersky, Adelyn Elizabeth", "Hunter, Luke Matthew", "Jaimes, Henry Theodore", "Khan, Emery", "Koch, Bates Kenneth", "Kulesza, Kahlan Rose", "Lewis, Beatrix", "Lopez, Samuel Robert", "Marotta, Ava", "McGown, Louie James", "Mintz, Florence Lorraine", "Robertson-Delaguardia, Dalton Ferris", "Robertson-Delaguardia, Rhys Stacey", "Sabo, Estelle", "Suddarth, Charles Landon", "Summers, Nora Christine"]
L = ["Aldrich, Drew Gabriel", "Anderson, Dahlia Louise Collins", "Buttitta, Oliver Rolfs", "Hunter, Luke Matthew", "Jaimes, Henry Theodore", "Kulesza, Kahlan Rose", "Lewis, Beatrix", "Lopez, Samuel Robert", "Mintz, Florence Lorraine", "Robertson-Delaguardia, Rhys Stacey", "Sabo, Estelle"]

counter = 0
for i in range(0,len(pagesCount.pages),pageCount,):
    output = PdfWriter()
    # output.append(inputpdf.pages[i])
    output.append(fileobj=inputpdf, pages=(i, i+pageCount))
    with open(L[counter]+" "+ sig +".pdf", "wb") as outputStream:
        output.write(outputStream)
    counter+=1

