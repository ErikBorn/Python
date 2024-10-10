from PyPDF2 import PdfReader, PdfWriter

filename = "hstranscripts.pdf"
sig = "Q1 Final"
pageCount = 1
inputpdf = open(filename, "rb")
pagesCount = PdfReader(open(filename, "rb"))

L = ["Bante, Madeline", "Bates, Vivian Louise", "Bhale, Saanvi Bhushan", "Blanchard, Marlowe Quinn", "Brown, Isabella Eun Hye", "Brown, Reese", "Burman, Alexandra", "Busch, Tatum", "Chambers, Devon Maeve", "Collins, Jacqueline Weston", "Crabtree, Josie", "Crysel, Caitlin Grace", "Davis, Alexandra Shilling", "Diaz, Marahi Fernanda", "Doyle, Trinity Marie", "Duenas, Regina", "Fowler, Tori Nicole", "Franciscus, Blair Madison", "Galindo, Thya", "Gonzalez, Ashley Helen", "Green, Natalie", "Hayward, Josie", "Hernandez, Andrea Soraya", "Hernandez, Isabel", "Holland, Reese", "Kim, Heesoo", "Kotowski, Kate", "Lane, Olivia Noelle", "Link, Olivia", "Martinez, Allyson", "Mathieson, Mykaela", "Misky, Ava", "Moore, Meghan Riley", "Mounsey, Mary Sarah", "Naranjo, Maria Valeria", "Neren, Rand Elizabeth", "Newton, Abigail Joy", "Nollsch, Janey Elizabeth", "Rayner, Isabel", "Reichert, Helen Anne", "Reyes, Hannah Elizabeth", "Rivas Quijada, Wendy Michelle", "Rivas, Nastasia", "Rodriguez, Emily", "Rothman, Eliana", "Shroff, Laila Montgomery", "Stehno, Julia", "Storch, Maria Barbara", "Talavera, Galya", "Taplin, Nell", "Tatar, Sydney", "Treuhaft, Rory", "Turner, Margaret", "Velez, Diana", "Waryn, Alexandra Katkin", "Werfelmann, Camille", "Winterling, Amelia"]

counter = 0
for i in range(0,len(pagesCount.pages),pageCount,):
    output = PdfWriter()
    output.append(fileobj=inputpdf, pages=(i, i+pageCount))
    with open(L[counter]+" "+ sig +".pdf", "wb") as outputStream:
        output.write(outputStream)
    counter+=1

