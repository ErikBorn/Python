from PyPDF2 import PdfReader, PdfWriter

filename = "hstranscripts.pdf"
sig = "Q1 Final"
pageCount = 1
inputpdf = open(filename, "rb")
pagesCount = PdfReader(open(filename, "rb"))

L = ["Amor, Janelle", "Anderson, Avery", "Anderson, Lauren", "Badillo Gomez, Valeria", "Barton, Kate Angeline", "Bauer, Matilda Grace", "Brower, Gabriella", "Capra, Tia Pauline", "Chavez Paz, Marlen", "Chism, Reese Honor", "Cooper, Helen Rose", "Cooper, Vienna Sue", "Darnell, Kenna'D Renee", "Drake, Pascale", "Draper, Maeve", "Dykes, Sawyer Amber Rose", "Engleby, Samantha Grace", "Fahey, Tatum", "Farnsworth, Grace", "Garcia, Abbi", "Garrett, Chloe", "Good, Annabelle Justine", "Higgins, Violet Gaines", "Hilario, Isabela", "Hunt, Beatrice Esther", "Idnani, Ava Grey", "Jacob, Reeve", "Jacoby, Heather Virginia", "Johnson, Audrey Claire", "Jones, Olivia", "Lindsay, Stella Lilian", "Lopez, Valeria", "McKinnon, Currie S", "Messerle, Lauren Reese", "Mills, Hazel", "Mills, Riley Ballantyne", "Miranda, Valeria Yareli", "Mueller, Kaitlin Elizabeth", "Nauman, Aleena", "O'Hara, Ally Elise", "Orrego, Sara Isabella", "Parra, Miranda", "Poore, Chloe Margaret", "Porter, Ellen Avery", "Rodriguez, Alondra", "Rodriguez, Jazlin Delilah", "Safavi, Ava Jacqueline", "Safavi, Darya Isabelle", "Safieddine, Sophia", "Salazar, Kendall Avery", "Seeber, Jasmine Angelina", "Sevilla, Jaqueline Mia", "Sheirbon, Kathleen", "Smith, Gabrielle Dawn", "Smith, Harper Rose", "Smith, Margaretta Marie", "Stander, Mia", "Stanley, Emily Bo", "Vernet, Emilia Jehanne Esther Marie", "Walker, Ella Sarah Reynolds", "Wester, Adriana Grace", "White, Eliana Susan", "Wysong, Paige Josephine", "Young, Averie"]

counter = 0
for i in range(0,len(pagesCount.pages),pageCount,):
    output = PdfWriter()
    output.append(fileobj=inputpdf, pages=(i, i+pageCount))
    with open(L[counter]+" "+ sig +".pdf", "wb") as outputStream:
        output.write(outputStream)
    counter+=1

