if __name__ == '__main__':
    totals = {2: 0, 4: 0, 8: 0, 16: 0, 24: 0}

    # f = open("mpi.txt", 'r')
    # for line in f.readlines():
    #     if line.startswith("== MPI"):
    #         current_bin = int(line.split(" ")[3])
    #     if line.startswith("Time"):
    #         totals[current_bin] = totals[current_bin] + float(line.split("  ")[1])

    # totals = { k: v/10 for k,v in totals.items() }
    # print(totals)

    # out = open("average_mpi.txt", 'w')
    # for k in sorted(totals):
    #     out.write("%f\n" % totals[k])

    # out.close()


    # totals = {2: 0, 4: 0, 8: 0, 16: 0, 24: 0}

    # f = open("openmp.txt", 'r')
    # for line in f.readlines():
    #     if line.startswith("== OpenMP"):
    #         current_bin = int(line.split(" ")[3])
    #     if line.startswith("Time"):
    #         totals[current_bin] = totals[current_bin] + float(line.split("  ")[1])

    # totals = { k: v/10 for k,v in totals.items() }
    # print(totals)

    # out = open("average_openmp.txt", 'w')
    # for k in sorted(totals):
    #     out.write("%f\n" % totals[k])

    # out.close()

    f = open("mpi_weak.txt", 'r')
    for line in f.readlines():
        if line.startswith("== MPI"):
            current_bin = int(line.split(" ")[3])
        if line.startswith("Time"):
            totals[current_bin] = totals[current_bin] + float(line.split("  ")[1])

    # totals = { k: v/10 for k,v in totals.items() }
    totals[2] = totals[2] / 10
    totals[4] = totals[4] / 10
    totals[8] = totals[8] / 10
    totals[16] = totals[16] / 10
    totals[24] = totals[24] / 7
    print(totals)

    out = open("average_mpi_weak.txt", 'w')
    for k in sorted(totals):
        out.write("%f\n" % totals[k])

    out.close()
