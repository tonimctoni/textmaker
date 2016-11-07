import re
contents=["" for i in range(5)]
#Read all files
for i in range(0,5):
    with open("01/%i.txt"%(i+1)) as f:
        contents[i]=f.read()

#Make it linux-like
for i in range(0,5): contents[i]=contents[i].replace("\r\n", "\n")
#Remove pagenumbers
for i in range(0,5): contents[i]=re.sub("""\s*Page \d+\s+""", """ """, contents[i])
for i in range(0,5): contents[i]=re.sub("""\n\s*.*?\d+\s*\n|\n\s*\d+.*?\s*\n""", """ """, contents[i])
#Some more cleanup
for i in range(0,5): contents[i]=contents[i].replace("GEORGE R.R. MARTIN", " ")
#Remove all breaklines ans multiple spaces
for i in range(0,5): contents[i]=re.sub("""\s+""", """ """, contents[i])
#Make it lower case
for i in range(0,5): contents[i]=contents[i].lower()
#Remove quotes
for i in range(0,5): contents[i]=contents[i].replace("\"", "");
#Improve ...s
for i in range(0,5): contents[i]=contents[i].replace(". . .", "...");

#Further cleanups
contents[2-1]=contents[2-1].replace("[1]", " ")
contents[3-1]=contents[3-1].replace("[on", "jon")
for i in range(0,5): contents[i]=contents[i].replace("~", "");
contents[1-1]=contents[1-1].replace("`", "t")
for i in range(0,5): contents[i]=contents[i].replace("]", "l");
for i in range(0,5): contents[i]=contents[i].replace("/", "");
for i in range(0,5): contents[i]=contents[i].replace("*", "");
for i in range(0,5): contents[i]=contents[i].replace("&", "g");

#Remove weird characters (unicode?)
allowed_chars="! ')(-,.103254769;:?acbedgfihkjmlonqpsrutwvyxz"#46
for i in range(0,5): contents[i]="".join([c for c in contents[i] if c in allowed_chars])


#Write all files back
for i in range(0,5):
    with open("02/%i.txt"%(i+1), "w") as f:
        f.write(contents[i])

#Also write a huge file
with open("02/concat.txt", "w") as f:
    for text in contents:
        f.write(text)

#Get set of characters
charset=set()
for text in contents:
    for c in text:
        charset.add(c)

print len(charset)
for c in charset:
    print c, ord(c)
print "".join(charset)
print len(charset)


# . => 1 character except newline
# r"" for a raw uninterpreted string 
# \w => 1 word character (A-Z,a-z,0-1,_,etc)
# \d => digits (0-9)
# \s => whitespace character (include tab, newline, etc)
# \S => non-whitespace
# + => one or more of the character to the left of it
# * => zero or more of the character to the left of it
# [] set of characters. Example: [\w.]+@[\w.]+ for e-mails

# (http(?:s)?\:\/\/[a-zA-Z0-9\-]+(?:\.[a-zA-Z0-9\-]+)*\.[a-zA-Z]{2,6}(?:\/?|(?:\/[\w\-]+)*)(?:\/?|\/\w+\.[a-zA-Z]{2,4}(?:\?[\w]+\=[\w\-]+)?)?(?:\&[\w]+\=[\w\-]+)*)
# url_re='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
