__kernel void contour( __global char* buftemp ,  __global int * result ,  int height,  int width ,int worksize )
{
    
    int piece = (width-2)/worksize;
    int step = width;
    
    int nch = 1;
    int nbd = 2;
    int deltas[16]={0};
    int prev_s , s, s_end;
    deltas[0] =  nch;
    deltas[1] = -step + nch;
    deltas[2] = -step;
    deltas[3] = -step - nch;
    deltas[4] = -nch;
    deltas[5] = step - nch;
    deltas[6] = step;
    deltas[7] = step + nch;
    __global char *i0 , *i1, *i3, *i4 = 0;
    
    
   
    deltas[8] =  nch;
    deltas[9] = -step + nch;
    deltas[10] = -step;
    deltas[11] = -step - nch;
    deltas[12] = -nch;
    deltas[13] = step - nch;
    deltas[14] = step;
    deltas[15] = step + nch;
    
     
    
    
    int contourcount = 0 ;
    int xdelta[9] = { 1, 1, 0, -1, -1, -1, 0, 1};
    int ydelta[9] = { 0, -1, -1, -1, 0, 1, 1, 1};
    
    int idx = get_global_id(0);
    
     
    for ( int y = 1 ; y < height-1; y++ ){
         for ( int x = piece*idx+1; x < piece*idx + piece; x++ ){
             
             bool b = buftemp[x + y * width]>0 && buftemp[x + y * width + 1] ==0 ;
             bool a = buftemp[x + y * width - 1]==0 && buftemp[x + y * width] == 1 ;
             
              
             int xcor=x-1;
             int ycor=y-1;
             if( a | b){
                 
                 
                 i0 = buftemp + x + y * width;
                 
                 if (a)
                    s_end = s = 4;
                 else
                     s_end = s = 0;
                     
                 do
                 {
                     s = (s - 1) & 7;
                     i1 = i0 + deltas[s];
                     if( *i1 != 0 )
                         break;
                 }
                 while( s != s_end );
                 if( s == s_end )
                 {
                     *i0 = (char) -nbd;
                     
                 }
                 else
                 {
                     i3 = i0;
                     prev_s = s ^ 4;
                     
                     for( ;; )
                     {
                         s_end = s;
                         
                         for( ;; )
                         {
                             i4 = i3 + deltas[++s];
                             if( *i4 != 0 )
                                 break;
                         }
                         s &= 7;
                         
                         
                         if( (unsigned) (s - 1) < (unsigned) s_end )
                         {
                             *i3 = (char) -nbd ;
                         }
                         else if( *i3 == 1 )
                         {
                             *i3 = nbd;
                         }
                         result[idx*50000+contourcount]=xcor;  //idx*50000+
                         result[idx*50000+contourcount + 1 ]= ycor;
                         contourcount+=2;
                         
                         xcor += xdelta[s];
                         ycor += ydelta[s];
                    
                         if( i4 == i0 && i3 == i1 )
                         {
                             
                             result[idx*50000+contourcount]=-1;
                              contourcount+=1;
                             break;   
                         }
                         i3 = i4;
                         s = (s + 4) & 7;
                             
                         }
                     }                       
                 
              
                 
             }                                                                            
    
        }
        } 
}
