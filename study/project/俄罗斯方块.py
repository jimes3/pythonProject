import pygame, sys, random

pygame.init()
pygame.display.set_caption("三角俄罗斯方块")

WKUAI, HKUAI = 10, 15
BIAN = 30
SIZE = WIDTH, HEIGHT = (WKUAI + 4) * BIAN, HKUAI * BIAN
WINDOW = pygame.display.set_mode(SIZE)
WINDOWRECT = WINDOW.get_rect()
NEXTWIN = pygame.rect.Rect(WKUAI * BIAN, 0, 4 * BIAN, HKUAI * BIAN)
CLOCK = pygame.time.Clock()
BLIST = [1, 2, 2, 3, 3, 4, 4, 5, 5]

def drawRect(x, y, color):
    pygame.draw.rect(WINDOW, color, (x * BIAN + 2, y * BIAN + 2, BIAN - 3, BIAN - 3), 1)

def drawSanJiao(x, y, color, g):
    # g 表示三角方向，1-4分别表示三角形直角朝 左下/右下/右上/左上
    if g == 1:
        pygame.draw.lines(WINDOW, color, True,
                          [(x * BIAN + 2, y * BIAN + 4), (x * BIAN + 2, y * BIAN + BIAN - 2), (x * BIAN + BIAN - 4, y * BIAN + BIAN - 2)])
    elif g == 2:
        pygame.draw.lines(WINDOW, color, True,
                          [(x * BIAN + BIAN - 2, y * BIAN + 4), (x * BIAN + BIAN - 2, y * BIAN + BIAN - 2), (x * BIAN + 4, y * BIAN + BIAN - 2)])
    elif g == 3:
        pygame.draw.lines(WINDOW, color, True,
                          [(x * BIAN + 3, y * BIAN + 2), (x * BIAN + BIAN - 2, y * BIAN + 2), (x * BIAN + BIAN - 2, y * BIAN + BIAN - 3)])
    elif g == 4:
        pygame.draw.lines(WINDOW, color, True,
                          [(x * BIAN + 2, y * BIAN + 2), (x * BIAN + BIAN - 4, y * BIAN + 2), (x * BIAN + 2, y * BIAN + BIAN - 4)])

def block(x, y, k, g):
    l = [[[] for _ in range(WKUAI)] for _ in range(HKUAI)]
    if k == 1:
        # 全三角
        if g == 1:
            l[y]    [x].append(3)
            l[y - 1][x].append(2)
            l[y]    [x + 1].append(4)
        elif g == 2:
            l[y]    [x].append(3)
            l[y - 1][x].append(2)
            l[y - 1][x + 1].append(1)
        elif g == 3:
            l[y - 1][x].append(2)
            l[y - 1][x + 1].append(1)
            l[y]    [x + 1].append(4)
        elif g == 4:
            l[y - 1][x + 1].append(1)
            l[y]    [x + 1].append(4)
            l[y]    [x].append(3)
    elif k == 2:
        # 左三角右正方形（三角直角朝右下）
        if g == 1:
            l[y]    [x].append(2)
            l[y]    [x + 1].append(500)
        elif g == 2:
            l[y]    [x].append(500)
            l[y - 1][x].append(1)
        elif g == 3:
            l[y]    [x].append(500)
            l[y]    [x + 1].append(4)
        elif g == 4:
            l[y]    [x].append(3)
            l[y - 1][x].append(500)
    elif k == 3:
        # 左正方形右三角（三角直角朝左下）
        if g == 1:
            l[y]    [x].append(500)
            l[y]    [x + 1].append(1)
        elif g == 2:
            l[y]    [x].append(4)
            l[y - 1][x].append(500)
        elif g == 3:
            l[y]    [x].append(3)
            l[y]    [x + 1].append(500)
        elif g == 4:
            l[y]    [x].append(500)
            l[y - 1][x].append(2)
    elif k == 4:
        # 三角
        if g == 1:
            l[y]    [x].append(2)
            l[y]    [x + 1].append(1)
        elif g == 2:
            l[y]    [x].append(4)
            l[y - 1][x].append(1)
        elif g == 3:
            l[y]    [x].append(3)
            l[y]    [x + 1].append(4)
        elif g == 4:
            l[y]    [x].append(3)
            l[y - 1][x].append(2)
    elif k == 5:
        # 大三角
        if g == 1:
            l[y]    [x].append(500)
            l[y - 1][x].append(1)
            l[y]    [x + 1].append(1)
        elif g == 2:
            l[y]    [x].append(4)
            l[y - 1][x].append(500)
            l[y - 1][x + 1].append(4)
        elif g == 3:
            l[y - 1][x + 1].append(500)
            l[y - 1][x].append(3)
            l[y]    [x + 1].append(3)
        elif g == 4:
            l[y]    [x].append(2)
            l[y]    [x + 1].append(500)
            l[y - 1][x + 1].append(2)
    return l

zhuangTaiList = [[[] for _ in range(WKUAI)] for _ in range(HKUAI)]
nzhuangTaiList = [[[] for _ in range(WKUAI)] for _ in range(HKUAI)]
nextZhuangTaiList = [[[] for _ in range(4)] for _ in range(HKUAI)]
llist = [[False for _ in range(WKUAI)] for _ in range(HKUAI)]
rlist = [[False for _ in range(WKUAI)] for _ in range(HKUAI)]
dlist = [[False for _ in range(WKUAI)] for _ in range(HKUAI)]

ox, oy = WKUAI // 2 - 1, 1
nx, ny = ox, oy

tt = 0
chongXin = False
shiBai = 0
nowZhuangTai = random.randint(1, 4)
nextZhuangTai = random.randint(1, 4)
fenShu = 0
font = pygame.font.Font("font/1.ttf", 16)
lm = rm = dm = False

drawNow = BLIST[random.randint(0, len(BLIST) - 1)]
drawNext = BLIST[random.randint(0, len(BLIST) - 1)]

zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)

while True:
    CLOCK.tick(60)

    WINDOW.fill((0, 0, 0))
    pygame.draw.line(WINDOW, (255, 255, 255), (WKUAI * BIAN, 0), (WKUAI * BIAN, HEIGHT), 1)
    pygame.draw.rect(WINDOW, (255, 255, 255), ((WKUAI + 1) * BIAN - 1, (HKUAI // 2 - 1) * BIAN - 1, BIAN * 2 + 3, BIAN * 2 + 3), 1)

    if not shiBai:
        # 失败判定
        for i in range(2):
            for j in range(ox, ox + 2):
                if nzhuangTaiList[i][j]:
                    shiBai = 1

        # 下一个
        nextZhuangTaiList = block(1, HKUAI // 2, drawNext, nextZhuangTai)
        r = -1000
        for i in range(HKUAI):
            for j in range(WKUAI):
                if zhuangTaiList[i][j]:
                    r = max(r, j)

    chongXin = False

    tt += 1
    if not dm and not shiBai:
        if tt >= 40:
            tt = 0
            ny += 1
            if ny == HKUAI:
                chongXin = True
            else:
                zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)
                for i in range(HKUAI):
                    for j in range(WKUAI):
                        if (zhuangTaiList[i][j] and dlist[i][j]) or \
                                (zhuangTaiList[i][j] and len(nzhuangTaiList[i][j]) and abs(zhuangTaiList[i][j][0] - nzhuangTaiList[i][j][0]) != 2):
                            ny -= 1
                            zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)
                            chongXin = True

            if chongXin:
                if not shiBai:
                    for i in range(HKUAI):
                        for j in range(WKUAI):
                            nzhuangTaiList[i][j] += zhuangTaiList[i][j]
                    nx, ny = ox, oy
                    drawNow = drawNext
                    drawNext = BLIST[random.randint(0, len(BLIST) - 1)]
                    nowZhuangTai = nextZhuangTai
                    nextZhuangTai = random.randint(1, 4)
                    zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)

            for i in range(HKUAI):
                xiao = True
                for j in range(WKUAI):
                    if not nzhuangTaiList[i][j] or (len(nzhuangTaiList[i][j]) == 1 and nzhuangTaiList[i][j][0] != 500):
                        xiao = False
                        break
                if xiao:
                    del nzhuangTaiList[i]
                    nzhuangTaiList.insert(0, [[] for _ in range(WKUAI)])
                    fenShu += 1

    for i in range(HKUAI):
        for j in range(WKUAI):
            if shiBai == 0:
                for k in zhuangTaiList[i][j]:
                    if k == 500:
                        drawRect(j, i, (255, 255, 255))
                    else:
                        drawSanJiao(j, i, (255, 255, 255), k)
            for k in nzhuangTaiList[i][j]:
                if k == 500:
                    drawRect(j, i, (0, 255, 0))
                else:
                    drawSanJiao(j, i, (0, 255, 0), k)
        for j in range(WKUAI, WKUAI + 4):
            for k in nextZhuangTaiList[i][j - WKUAI]:
                if k == 500:
                    drawRect(j, i, (255, 255, 0))
                else:
                    drawSanJiao(j, i, (255, 255, 0), k)

    text = font.render("Score: " + str(fenShu), True, (255, 255, 255))
    textRect = text.get_rect()
    textRect.center = NEXTWIN.center
    textRect.y += int(BIAN * 1.5)
    WINDOW.blit(text, textRect)
    if shiBai:
        text = font.render("Game Over~", True, (255, 255, 0))
        textRect = text.get_rect()
        textRect.center = WINDOWRECT.center
        textRect.y -= 20
        WINDOW.blit(text, textRect)
        text2 = font.render("Your Score: " + str(fenShu), True, (255, 255, 0))
        textRect2 = text2.get_rect()
        textRect2.center = WINDOWRECT.center
        WINDOW.blit(text2, textRect2)
        text3 = font.render("Press C To Start Over~", True, (255, 255, 0))
        textRect3 = text3.get_rect()
        textRect3.center = WINDOWRECT.center
        textRect3.y += 20
        WINDOW.blit(text3, textRect3)

    llist = [[False for _ in range(WKUAI)] for _ in range(HKUAI)]
    rlist = [[False for _ in range(WKUAI)] for _ in range(HKUAI)]
    dlist = [[False for _ in range(WKUAI)] for _ in range(HKUAI)]

    for i in range(HKUAI):
        for j in range(WKUAI):
            if nzhuangTaiList[i][j]:
                # down l
                if 500 in nzhuangTaiList[i][j] or 4 in nzhuangTaiList[i][j] or 3 in nzhuangTaiList[i][j]:
                    dlist[i][j] = True
                if (1 in nzhuangTaiList[i][j] or 2 in nzhuangTaiList[i][j]) and i + 1 < HKUAI:
                    dlist[i + 1][j] = True
                # left l
                if 500 in nzhuangTaiList[i][j] or 2 in nzhuangTaiList[i][j] or 3 in nzhuangTaiList[i][j]:
                    llist[i][j] = True
                if (1 in nzhuangTaiList[i][j] or 4 in nzhuangTaiList[i][j]) and j - 1 != -1:
                    llist[i][j - 1] = True
                # right l
                if 500 in nzhuangTaiList[i][j] or 1 in nzhuangTaiList[i][j] or 4 in nzhuangTaiList[i][j]:
                    rlist[i][j] = True
                if (2 in nzhuangTaiList[i][j] or 3 in nzhuangTaiList[i][j]) and j + 1 != WKUAI:
                    rlist[i][j + 1] = True

    if tt % 5 == 0:
        if lm:
            if nx > 0 and not shiBai:
                nx -= 1
                zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)
                bt = False
                for i in range(HKUAI):
                    if bt:
                        break
                    for j in range(WKUAI):
                        if (zhuangTaiList[i][j] and llist[i][j]) or \
                                (zhuangTaiList[i][j] and len(nzhuangTaiList[i][j]) and abs(zhuangTaiList[i][j][0] - nzhuangTaiList[i][j][0]) != 2):
                            bt = True
                            break
                if bt:
                    nx += 1
                    zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)

        elif rm:
            if r < WKUAI - 1 and not shiBai:
                nx += 1
                zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)
                bt = False
                for i in range(HKUAI):
                    if bt:
                        break
                    for j in range(WKUAI):
                        if (zhuangTaiList[i][j] and rlist[i][j]) or \
                                (zhuangTaiList[i][j] and len(nzhuangTaiList[i][j]) and abs(zhuangTaiList[i][j][0] - nzhuangTaiList[i][j][0]) != 2):
                            bt = True
                            break
                if bt:
                    nx -= 1
                    zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)

        if dm:
            if ny < HKUAI - 1 and not shiBai:
                ny += 1
                zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)
                bt = False
                for i in range(HKUAI):
                    if bt:
                        break
                    for j in range(WKUAI):
                        if (zhuangTaiList[i][j] and dlist[i][j]) or \
                                (zhuangTaiList[i][j] and len(nzhuangTaiList[i][j]) and abs(zhuangTaiList[i][j][0] - nzhuangTaiList[i][j][0]) != 2):
                            bt = True
                            break
                if bt:
                    ny -= 1
                    zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)
                    dm = False
            else:
                dm = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_DOWN:
                lm = True if event.key == pygame.K_LEFT else lm
                rm = True if event.key == pygame.K_RIGHT else rm
                dm = True if event.key == pygame.K_DOWN else dm

            elif event.key == pygame.K_SPACE:
                if nx <= WKUAI - 2:
                    nowZhuangTai += 1
                    if nowZhuangTai == 5:
                        nowZhuangTai = 1
                    zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)
                    bt = False
                    for i in range(HKUAI):
                        if bt:
                            break
                        for j in range(WKUAI):
                            if zhuangTaiList[i][j] and nzhuangTaiList[i][j] and zhuangTaiList[i][j][0] - nzhuangTaiList[i][j][0] != 2:
                                nowZhuangTai -= 1
                                if nowZhuangTai == 0:
                                    nowZhuangTai = 4
                                zhuangTaiList = block(nx, ny, drawNow, nowZhuangTai)
                                bt = True
                                break

            elif event.key == pygame.K_c:
                if shiBai != 0:
                    zhuangTaiList = [[[] for _ in range(WKUAI)] for _ in range(HKUAI)]
                    nzhuangTaiList = [[[] for _ in range(WKUAI)] for _ in range(HKUAI)]
                    nextZhuangTaiList = [[[] for _ in range(4)] for _ in range(HKUAI)]
                    llist = [[False for _ in range(WKUAI)] for _ in range(HKUAI)]
                    rlist = [[False for _ in range(WKUAI)] for _ in range(HKUAI)]
                    dlist = [[False for _ in range(WKUAI)] for _ in range(HKUAI)]

                    nx, ny = ox, oy

                    tt = 0
                    chongXin = False
                    shiBai = 0
                    nowZhuangTai = random.randint(1, 4)
                    nextZhuangTai = random.randint(1, 4)
                    fenShu = 0
                    font = pygame.font.Font("font/1.ttf", 16)
                    lm = False
                    rm = False
                    dm = False

                    drawNow = BLIST[random.randint(0, len(BLIST) - 1)]
                    drawNext = BLIST[random.randint(0, len(BLIST) - 1)]

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_DOWN:
                lm = False if event.key == pygame.K_LEFT else lm
                rm = False if event.key == pygame.K_RIGHT else rm
                dm = False if event.key == pygame.K_DOWN else dm

    pygame.display.update()