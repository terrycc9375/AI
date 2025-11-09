import os

from sympy import content

# grading rule
'''
error < 10^-5 => 100
error < 10^-4 => 90
error < 10^-3 => 80
error < 10^-2 => 0

the result is in result.log with multiple lines
the total score is the avg score of each line
'''

def score(error):
    if error < 1e-5:
        return 100
    elif error < 1e-4:
        return 90
    elif error < 1e-3:
        return 80
    elif error < 1e-2:
        return 0
    else:
        return 0

def main ():
    total_score = 0
    count = 0
    # with open('result.log', 'rb') as f_in:
    #     content = f_in.read().decode('utf-8-sig')
    # with open('result.log', 'w', encoding='utf-8') as f_out:
    #     f_out.write(content)
    with open('result.log', 'r', encoding='utf-16-le') as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            try:
                error = float(line.strip().replace('\ufeff', ''))
                s = score(error)
                total_score += s
                count += 1
                print(f'Error: {error}, Score: {s}')
            except ValueError as e:
                print(f"Invalid: {e}")
                continue
    
    if count > 0:
        avg_score = total_score / count
        print(f'Average score: {avg_score}')

if __name__ == '__main__':
    main()
