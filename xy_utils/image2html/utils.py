import math

def writeHTML(file_name, im_paths, captions, height=200, width=200):
    f=open(file_name, 'w')
    html=[]
    f.write('<!DOCTYPE html>\n')
    f.write('<html><body>\n')
    f.write('<table>\n')
    for row in range(len(im_paths)):
        f.write('<tr>\n')
        for col in range(len(im_paths[row])):
            f.write('<td>')
            f.write(captions[row][col])
            f.write('</td>')
            f.write('    ')
        f.write('\n</tr>\n')

        f.write('<tr>\n')
        for col in range(len(im_paths[row])):
            f.write('<td><img src="')
            f.write(im_paths[row][col])
            f.write('" height='+str(height)+' width='+str(width)+'"/></td>')
            f.write('    ')
        f.write('\n</tr>\n')
        f.write('<p></p>')
    f.write('</table>\n')
    f.close()

def writeSeqHTML(file_name, im_paths, captions, col_n, height=200, width=200):
    total_n = len(im_paths)
    row_n = int(math.ceil(float(total_n) / col_n))
    f=open(file_name, 'w')
    html=[]
    f.write('<!DOCTYPE html>\n')
    f.write('<html><body>\n')
    f.write('<table>\n')
    for row in range(row_n):
        base_count = row * col_n
        f.write('<tr>\n')
        for col in range(col_n):
            if base_count + col < total_n:
                f.write('<td>')
                f.write(captions[base_count + col])
                f.write('</td>')
                f.write('    ')
        f.write('\n</tr>\n')

        f.write('<tr>\n')
        for col in range(col_n):
            if base_count + col < total_n:
                f.write('<td><img src="')
                f.write(im_paths[base_count + col])
                f.write('" height='+str(height)+' width='+str(width)+'"/></td>')
                f.write('    ')
        f.write('\n</tr>\n')
        f.write('<p></p>')
    f.write('</table>\n')
    f.close()