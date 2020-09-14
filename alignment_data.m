function [A_alignment] = alignment_data(A)

    L = length(A);
    window = L/3;
    for i = 1 : L-window
        total_E = 0;
        for nn = i : i + window
            E{nn}=A(nn)*A(nn);
            total_E = total_E + E{nn};
        end
        E_win{i} = total_E;
    end
    max = 0 ;
    max_i = 0 ;
    for i = 1 : L-window

        if (E_win{i}>=max)
            max = E_win{i};
            max_i = i;
         else
            max = max;
            max_i = max_i;  
        end
      
    end
    ind = max_i;
    A_alignment = A(ind : ind + window-1);
end