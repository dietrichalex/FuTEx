function x_new = getStateRect(x_old,dt, v_const)

x_new = x_old;
numSteps = 10;

for i=1:numSteps
  x_new = getStateRectIntegrate(x_new,dt/numSteps, v_const);
end


function x_new = getStateRectIntegrate(x_old,dt, v_const)

ymin = -30;
ymax = 30;
zmin = 20;
zmax = 70;


LB = 1;
RB = 2;
RT = 3;
LT = 4;

MB = 5;
MR = 6;
MT = 7;
ML = 8;
persistent last_state;
if(isempty(last_state) || isempty(x_old))
  last_state = -1;
end


if(nargin>2 && ~isempty(v_const) && v_const > 0)
  a = 0;
else
  a = 90;
  v_const = 0;
end

if(isempty(x_old))
  y0 = ymin;
  z0 = zmin;
else
  y0 = x_old(1);
  z0 = x_old(2);
  vy = x_old(3);
  vz = x_old(4);
  ay = x_old(5);
  az = x_old(6);
end

d = 0.1;

%right bottom
if (abs(y0-ymax)<d && abs(z0-zmin)<d)
  ynew = ymax;
  znew = zmin;  
  state = RB;
  % middle right
elseif (abs(y0-ymax)<d && abs(z0-(zmax+zmin)/2)<d)
  ynew = ymax;
  znew = (zmax+zmin)/2;    
  state = MR;
  %right top
elseif (abs(y0-ymax)<d && abs(z0-zmax)<d)
  ynew = ymax;
  znew = zmax;    
  state = RT;
  %middle top
elseif (abs(z0-zmax)<d && abs(y0-(ymax+ymin)/2)<d)
  ynew = (ymax+ymin)/2;        
  znew = zmax;
  state = MT;
  %left top
elseif(abs(y0-ymin)<d && abs(z0-zmax)<d)
  ynew = ymin;
  znew = zmax;     
  state = LT;
  %middle left
elseif (abs(y0-ymin)<d && abs(z0-(zmax+zmin)/2)<d)
  ynew = ymin;
  znew = (zmax+zmin)/2;    
  state = ML;
  %left bottom
elseif (abs(y0-ymin)<d && abs(z0-zmin)<d)
  ynew = ymin;
  znew = zmin;     
  state = LB;
  % middle bottom
elseif (abs(z0-zmin)<d && abs(y0-(ymax+ymin)/2)<d)
  ynew = (ymax+ymin)/2;
  znew = zmin;   
  state = MB;
else
  state = last_state;
end



if(last_state ~= state)
  last_state = state;

  switch(state)
    case(MB)
      ay = -ay;
    case(MR)
      az = -az;
    case(MT)
      ay = -ay;
    case(ML)
      az = -az;
    case(RB)
      if(v_const>0)
        vz = v_const;
        az = 0;
      else
        vz = 0;
        az = a;
      end
      vy = 0;
      ay = 0;

    case(RT)
      if(v_const>0)
        vy = -v_const;
        ay = 0;
      else
        vy = 0;
        ay = -a;
      end
      vz = 0;
      az = 0;

    case(LT)
      if(v_const>0)
        vz = -v_const;
        az = 0;
      else
        vz = 0;
        az = -a;
      end
      vy = 0;
      ay = 0;
    case(LB)
      if(v_const>0)
        vy = v_const;
        ay = 0;
      else
        vy = 0;
        ay = a;
      end
      vz = 0;
      az = 0;

  end
  x_new = [ynew, znew, vy, vz, ay, az ]';
else
  vy_new = vy+ay*dt;
  vz_new = vz+az*dt;
  x_new = [y0+vy*dt+0.5*dt^2*ay, z0+vz*dt+0.5*az*dt^2, vy_new, vz_new, ay, az ]';  
end




