��
�
�

B
AssignVariableOp
resource
value"dtype"
dtypetype�
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
�
.update_targets/periodic_update_targets/counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *?
shared_name0.update_targets/periodic_update_targets/counter
�
Bupdate_targets/periodic_update_targets/counter/Read/ReadVariableOpReadVariableOp.update_targets/periodic_update_targets/counter*
_output_shapes
: *
dtype0	
�
QNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameQNetwork/dense_2/kernel
�
+QNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_2/kernel*
_output_shapes

:*
dtype0
�
QNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameQNetwork/dense_2/bias
{
)QNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_2/bias*
_output_shapes
:*
dtype0
�
%QNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*6
shared_name'%QNetwork/EncodingNetwork/dense/kernel
�
9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense/kernel*
_output_shapes

:	*
dtype0
�
#QNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#QNetwork/EncodingNetwork/dense/bias
�
7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp#QNetwork/EncodingNetwork/dense/bias*
_output_shapes
:*
dtype0
�
'QNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'QNetwork/EncodingNetwork/dense_1/kernel
�
;QNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/dense_1/kernel*
_output_shapes

:*
dtype0
�
%QNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%QNetwork/EncodingNetwork/dense_1/bias
�
9QNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense_1/bias*
_output_shapes
:*
dtype0
�
TargetQNetwork/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameTargetQNetwork/dense_5/kernel
�
1TargetQNetwork/dense_5/kernel/Read/ReadVariableOpReadVariableOpTargetQNetwork/dense_5/kernel*
_output_shapes

:*
dtype0
�
TargetQNetwork/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameTargetQNetwork/dense_5/bias
�
/TargetQNetwork/dense_5/bias/Read/ReadVariableOpReadVariableOpTargetQNetwork/dense_5/bias*
_output_shapes
:*
dtype0
�
-TargetQNetwork/EncodingNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*>
shared_name/-TargetQNetwork/EncodingNetwork/dense_3/kernel
�
ATargetQNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOp-TargetQNetwork/EncodingNetwork/dense_3/kernel*
_output_shapes

:	*
dtype0
�
+TargetQNetwork/EncodingNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+TargetQNetwork/EncodingNetwork/dense_3/bias
�
?TargetQNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOp+TargetQNetwork/EncodingNetwork/dense_3/bias*
_output_shapes
:*
dtype0
�
-TargetQNetwork/EncodingNetwork/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*>
shared_name/-TargetQNetwork/EncodingNetwork/dense_4/kernel
�
ATargetQNetwork/EncodingNetwork/dense_4/kernel/Read/ReadVariableOpReadVariableOp-TargetQNetwork/EncodingNetwork/dense_4/kernel*
_output_shapes

:*
dtype0
�
+TargetQNetwork/EncodingNetwork/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+TargetQNetwork/EncodingNetwork/dense_4/bias
�
?TargetQNetwork/EncodingNetwork/dense_4/bias/Read/ReadVariableOpReadVariableOp+TargetQNetwork/EncodingNetwork/dense_4/bias*
_output_shapes
:*
dtype0
�
#RMSprop/QNetwork/dense_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#RMSprop/QNetwork/dense_2/kernel/rms
�
7RMSprop/QNetwork/dense_2/kernel/rms/Read/ReadVariableOpReadVariableOp#RMSprop/QNetwork/dense_2/kernel/rms*
_output_shapes

:*
dtype0
�
!RMSprop/QNetwork/dense_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!RMSprop/QNetwork/dense_2/bias/rms
�
5RMSprop/QNetwork/dense_2/bias/rms/Read/ReadVariableOpReadVariableOp!RMSprop/QNetwork/dense_2/bias/rms*
_output_shapes
:*
dtype0
�
1RMSprop/QNetwork/EncodingNetwork/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*B
shared_name31RMSprop/QNetwork/EncodingNetwork/dense/kernel/rms
�
ERMSprop/QNetwork/EncodingNetwork/dense/kernel/rms/Read/ReadVariableOpReadVariableOp1RMSprop/QNetwork/EncodingNetwork/dense/kernel/rms*
_output_shapes

:	*
dtype0
�
/RMSprop/QNetwork/EncodingNetwork/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/RMSprop/QNetwork/EncodingNetwork/dense/bias/rms
�
CRMSprop/QNetwork/EncodingNetwork/dense/bias/rms/Read/ReadVariableOpReadVariableOp/RMSprop/QNetwork/EncodingNetwork/dense/bias/rms*
_output_shapes
:*
dtype0
�
3RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*D
shared_name53RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/rms
�
GRMSprop/QNetwork/EncodingNetwork/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOp3RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/rms*
_output_shapes

:*
dtype0
�
1RMSprop/QNetwork/EncodingNetwork/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31RMSprop/QNetwork/EncodingNetwork/dense_1/bias/rms
�
ERMSprop/QNetwork/EncodingNetwork/dense_1/bias/rms/Read/ReadVariableOpReadVariableOp1RMSprop/QNetwork/EncodingNetwork/dense_1/bias/rms*
_output_shapes
:*
dtype0
�
"RMSprop/QNetwork/dense_2/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"RMSprop/QNetwork/dense_2/kernel/mg
�
6RMSprop/QNetwork/dense_2/kernel/mg/Read/ReadVariableOpReadVariableOp"RMSprop/QNetwork/dense_2/kernel/mg*
_output_shapes

:*
dtype0
�
 RMSprop/QNetwork/dense_2/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" RMSprop/QNetwork/dense_2/bias/mg
�
4RMSprop/QNetwork/dense_2/bias/mg/Read/ReadVariableOpReadVariableOp RMSprop/QNetwork/dense_2/bias/mg*
_output_shapes
:*
dtype0
�
0RMSprop/QNetwork/EncodingNetwork/dense/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*A
shared_name20RMSprop/QNetwork/EncodingNetwork/dense/kernel/mg
�
DRMSprop/QNetwork/EncodingNetwork/dense/kernel/mg/Read/ReadVariableOpReadVariableOp0RMSprop/QNetwork/EncodingNetwork/dense/kernel/mg*
_output_shapes

:	*
dtype0
�
.RMSprop/QNetwork/EncodingNetwork/dense/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.RMSprop/QNetwork/EncodingNetwork/dense/bias/mg
�
BRMSprop/QNetwork/EncodingNetwork/dense/bias/mg/Read/ReadVariableOpReadVariableOp.RMSprop/QNetwork/EncodingNetwork/dense/bias/mg*
_output_shapes
:*
dtype0
�
2RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*C
shared_name42RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/mg
�
FRMSprop/QNetwork/EncodingNetwork/dense_1/kernel/mg/Read/ReadVariableOpReadVariableOp2RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/mg*
_output_shapes

:*
dtype0
�
0RMSprop/QNetwork/EncodingNetwork/dense_1/bias/mgVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20RMSprop/QNetwork/EncodingNetwork/dense_1/bias/mg
�
DRMSprop/QNetwork/EncodingNetwork/dense_1/bias/mg/Read/ReadVariableOpReadVariableOp0RMSprop/QNetwork/EncodingNetwork/dense_1/bias/mg*
_output_shapes
:*
dtype0

NoOpNoOp
�D
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�C
value�CB�C B�C
�

_q_network
_target_q_network

_optimizer
_update_target
_target_greedy_policy
_policy
_collect_policy
_collect_data_context
	_data_context

_train_argspec
_train_step_counter
_as_transition

signatures
t
_encoder
_q_value_layer
regularization_losses
	variables
trainable_variables
	keras_api
t
_encoder
_q_value_layer
regularization_losses
	variables
trainable_variables
	keras_api
�
iter
	decay
learning_rate
momentum
rho
)rms�
*rms�
/rms�
0rms�
1rms�
2rms�	)mg�	*mg�	/mg�	0mg�	1mg�	2mg�

_counter

 _wrapped_policy

!_wrapped_policy
(
"_greedy_policy
#_random_policy
 
 
 
LJ
VARIABLE_VALUEVariable._train_step_counter/.ATTRIBUTES/VARIABLE_VALUE

	_data_context
 
n
$_postprocessing_layers
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
 
*
/0
01
12
23
)4
*5
*
/0
01
12
23
)4
*5
�
regularization_losses
3metrics
	variables
4layer_metrics
5non_trainable_variables
trainable_variables

6layers
7layer_regularization_losses
n
8_postprocessing_layers
9regularization_losses
:	variables
;trainable_variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
 
*
C0
D1
E2
F3
=4
>5
*
C0
D1
E2
F3
=4
>5
�
regularization_losses
Gmetrics
	variables
Hlayer_metrics
Inon_trainable_variables
trainable_variables

Jlayers
Klayer_regularization_losses
LJ
VARIABLE_VALUERMSprop/iter*_optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUERMSprop/decay+_optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUERMSprop/learning_rate3_optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUERMSprop/momentum._optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUERMSprop/rho)_optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE.update_targets/periodic_update_targets/counter2_update_target/_counter/.ATTRIBUTES/VARIABLE_VALUE


_q_network


_q_network

!_wrapped_policy
 

L0
M1
N2
 

/0
01
12
23

/0
01
12
23
�
%regularization_losses
Ometrics
&	variables
Player_metrics
Qnon_trainable_variables
'trainable_variables

Rlayers
Slayer_regularization_losses
hf
VARIABLE_VALUEQNetwork/dense_2/kernel;_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEQNetwork/dense_2/bias9_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
�
+regularization_losses
Tmetrics
,	variables
Ulayer_metrics
Vnon_trainable_variables
-trainable_variables

Wlayers
Xlayer_regularization_losses
lj
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense/kernel1_q_network/variables/0/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#QNetwork/EncodingNetwork/dense/bias1_q_network/variables/1/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'QNetwork/EncodingNetwork/dense_1/kernel1_q_network/variables/2/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense_1/bias1_q_network/variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
 

Y0
Z1
[2
 

C0
D1
E2
F3

C0
D1
E2
F3
�
9regularization_losses
\metrics
:	variables
]layer_metrics
^non_trainable_variables
;trainable_variables

_layers
`layer_regularization_losses
us
VARIABLE_VALUETargetQNetwork/dense_5/kernelB_target_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUETargetQNetwork/dense_5/bias@_target_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
�
?regularization_losses
ametrics
@	variables
blayer_metrics
cnon_trainable_variables
Atrainable_variables

dlayers
elayer_regularization_losses
{y
VARIABLE_VALUE-TargetQNetwork/EncodingNetwork/dense_3/kernel8_target_q_network/variables/0/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE+TargetQNetwork/EncodingNetwork/dense_3/bias8_target_q_network/variables/1/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE-TargetQNetwork/EncodingNetwork/dense_4/kernel8_target_q_network/variables/2/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE+TargetQNetwork/EncodingNetwork/dense_4/bias8_target_q_network/variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
 
R
fregularization_losses
g	variables
htrainable_variables
i	keras_api
h

/kernel
0bias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
h

1kernel
2bias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
 
 
 

L0
M1
N2
 
 
 
 
 
 
R
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
h

Ckernel
Dbias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
h

Ekernel
Fbias
zregularization_losses
{	variables
|trainable_variables
}	keras_api
 
 
 

Y0
Z1
[2
 
 
 
 
 
 
 
 
 
�
fregularization_losses
~metrics
g	variables
layer_metrics
�non_trainable_variables
htrainable_variables
�layers
 �layer_regularization_losses
 

/0
01

/0
01
�
jregularization_losses
�metrics
k	variables
�layer_metrics
�non_trainable_variables
ltrainable_variables
�layers
 �layer_regularization_losses
 

10
21

10
21
�
nregularization_losses
�metrics
o	variables
�layer_metrics
�non_trainable_variables
ptrainable_variables
�layers
 �layer_regularization_losses
 
 
 
�
rregularization_losses
�metrics
s	variables
�layer_metrics
�non_trainable_variables
ttrainable_variables
�layers
 �layer_regularization_losses
 

C0
D1

C0
D1
�
vregularization_losses
�metrics
w	variables
�layer_metrics
�non_trainable_variables
xtrainable_variables
�layers
 �layer_regularization_losses
 

E0
F1

E0
F1
�
zregularization_losses
�metrics
{	variables
�layer_metrics
�non_trainable_variables
|trainable_variables
�layers
 �layer_regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
��
VARIABLE_VALUE#RMSprop/QNetwork/dense_2/kernel/rmsZ_q_network/_q_value_layer/kernel/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!RMSprop/QNetwork/dense_2/bias/rmsX_q_network/_q_value_layer/bias/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE1RMSprop/QNetwork/EncodingNetwork/dense/kernel/rmsP_q_network/variables/0/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE/RMSprop/QNetwork/EncodingNetwork/dense/bias/rmsP_q_network/variables/1/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE3RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/rmsP_q_network/variables/2/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE1RMSprop/QNetwork/EncodingNetwork/dense_1/bias/rmsP_q_network/variables/3/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"RMSprop/QNetwork/dense_2/kernel/mgY_q_network/_q_value_layer/kernel/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE RMSprop/QNetwork/dense_2/bias/mgW_q_network/_q_value_layer/bias/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0RMSprop/QNetwork/EncodingNetwork/dense/kernel/mgO_q_network/variables/0/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE.RMSprop/QNetwork/EncodingNetwork/dense/bias/mgO_q_network/variables/1/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/mgO_q_network/variables/2/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE0RMSprop/QNetwork/EncodingNetwork/dense_1/bias/mgO_q_network/variables/3/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUE
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOpBupdate_targets/periodic_update_targets/counter/Read/ReadVariableOp+QNetwork/dense_2/kernel/Read/ReadVariableOp)QNetwork/dense_2/bias/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOp7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOp;QNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOp9QNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOp1TargetQNetwork/dense_5/kernel/Read/ReadVariableOp/TargetQNetwork/dense_5/bias/Read/ReadVariableOpATargetQNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOp?TargetQNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpATargetQNetwork/EncodingNetwork/dense_4/kernel/Read/ReadVariableOp?TargetQNetwork/EncodingNetwork/dense_4/bias/Read/ReadVariableOp7RMSprop/QNetwork/dense_2/kernel/rms/Read/ReadVariableOp5RMSprop/QNetwork/dense_2/bias/rms/Read/ReadVariableOpERMSprop/QNetwork/EncodingNetwork/dense/kernel/rms/Read/ReadVariableOpCRMSprop/QNetwork/EncodingNetwork/dense/bias/rms/Read/ReadVariableOpGRMSprop/QNetwork/EncodingNetwork/dense_1/kernel/rms/Read/ReadVariableOpERMSprop/QNetwork/EncodingNetwork/dense_1/bias/rms/Read/ReadVariableOp6RMSprop/QNetwork/dense_2/kernel/mg/Read/ReadVariableOp4RMSprop/QNetwork/dense_2/bias/mg/Read/ReadVariableOpDRMSprop/QNetwork/EncodingNetwork/dense/kernel/mg/Read/ReadVariableOpBRMSprop/QNetwork/EncodingNetwork/dense/bias/mg/Read/ReadVariableOpFRMSprop/QNetwork/EncodingNetwork/dense_1/kernel/mg/Read/ReadVariableOpDRMSprop/QNetwork/EncodingNetwork/dense_1/bias/mg/Read/ReadVariableOpConst*,
Tin%
#2!		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_48585169
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameVariableRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rho.update_targets/periodic_update_targets/counterQNetwork/dense_2/kernelQNetwork/dense_2/bias%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/bias'QNetwork/EncodingNetwork/dense_1/kernel%QNetwork/EncodingNetwork/dense_1/biasTargetQNetwork/dense_5/kernelTargetQNetwork/dense_5/bias-TargetQNetwork/EncodingNetwork/dense_3/kernel+TargetQNetwork/EncodingNetwork/dense_3/bias-TargetQNetwork/EncodingNetwork/dense_4/kernel+TargetQNetwork/EncodingNetwork/dense_4/bias#RMSprop/QNetwork/dense_2/kernel/rms!RMSprop/QNetwork/dense_2/bias/rms1RMSprop/QNetwork/EncodingNetwork/dense/kernel/rms/RMSprop/QNetwork/EncodingNetwork/dense/bias/rms3RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/rms1RMSprop/QNetwork/EncodingNetwork/dense_1/bias/rms"RMSprop/QNetwork/dense_2/kernel/mg RMSprop/QNetwork/dense_2/bias/mg0RMSprop/QNetwork/EncodingNetwork/dense/kernel/mg.RMSprop/QNetwork/EncodingNetwork/dense/bias/mg2RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/mg0RMSprop/QNetwork/EncodingNetwork/dense_1/bias/mg*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_48585272��
�
�
cond_false_48585011
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_placeholder_4
cond_placeholder_5
cond_placeholder_6
cond_placeholder_7
cond_placeholder_8
cond_placeholder_9
cond_placeholder_10
cond_placeholder_11
cond_identity_equal

cond_identity_1
4
	cond/NoOpNoOp*
_output_shapes
 2
	cond/NoOpl
cond/IdentityIdentitycond_identity_equal
^cond/NoOp*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*E
_input_shapes4
2::::::::::::: :

_output_shapes
: 
�
�
foldr_while_body_48584252(
$foldr_while_foldr_while_loop_counter.
*foldr_while_foldr_while_maximum_iterations
foldr_while_placeholder
foldr_while_placeholder_1c
_foldr_while_tensorarrayv2read_tensorlistgetitem_foldr_tensorarrayunstack_tensorlistfromtensor_0g
cfoldr_while_tensorarrayv2read_1_tensorlistgetitem_foldr_tensorarrayunstack_1_tensorlistfromtensor_0
foldr_while_identity
foldr_while_identity_1
foldr_while_identity_2
foldr_while_identity_3a
]foldr_while_tensorarrayv2read_tensorlistgetitem_foldr_tensorarrayunstack_tensorlistfromtensore
afoldr_while_tensorarrayv2read_1_tensorlistgetitem_foldr_tensorarrayunstack_1_tensorlistfromtensorh
foldr/while/sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
foldr/while/sub/y
foldr/while/subSubfoldr_while_placeholderfoldr/while/sub/y:output:0*
T0*
_output_shapes
: 2
foldr/while/sub�
=foldr/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:�2?
=foldr/while/TensorArrayV2Read/TensorListGetItem/element_shape�
/foldr/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem_foldr_while_tensorarrayv2read_tensorlistgetitem_foldr_tensorarrayunstack_tensorlistfromtensor_0foldr/while/sub:z:0Ffoldr/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes	
:�*
element_dtype021
/foldr/while/TensorArrayV2Read/TensorListGetItem�
?foldr/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:�2A
?foldr/while/TensorArrayV2Read_1/TensorListGetItem/element_shape�
1foldr/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemcfoldr_while_tensorarrayv2read_1_tensorlistgetitem_foldr_tensorarrayunstack_1_tensorlistfromtensor_0foldr/while/sub:z:0Hfoldr/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*
_output_shapes	
:�*
element_dtype023
1foldr/while/TensorArrayV2Read_1/TensorListGetItem�
foldr/while/mulMulfoldr_while_placeholder_18foldr/while/TensorArrayV2Read_1/TensorListGetItem:item:0*
T0*
_output_shapes	
:�2
foldr/while/mul�
foldr/while/addAddV2foldr/while/mul:z:06foldr/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes	
:�2
foldr/while/addl
foldr/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
foldr/while/add_1/y�
foldr/while/add_1AddV2$foldr_while_foldr_while_loop_counterfoldr/while/add_1/y:output:0*
T0*
_output_shapes
: 2
foldr/while/add_1p
foldr/while/IdentityIdentityfoldr/while/add_1:z:0*
T0*
_output_shapes
: 2
foldr/while/Identity�
foldr/while/Identity_1Identity*foldr_while_foldr_while_maximum_iterations*
T0*
_output_shapes
: 2
foldr/while/Identity_1r
foldr/while/Identity_2Identityfoldr/while/sub:z:0*
T0*
_output_shapes
: 2
foldr/while/Identity_2w
foldr/while/Identity_3Identityfoldr/while/add:z:0*
T0*
_output_shapes	
:�2
foldr/while/Identity_3"5
foldr_while_identityfoldr/while/Identity:output:0"9
foldr_while_identity_1foldr/while/Identity_1:output:0"9
foldr_while_identity_2foldr/while/Identity_2:output:0"9
foldr_while_identity_3foldr/while/Identity_3:output:0"�
afoldr_while_tensorarrayv2read_1_tensorlistgetitem_foldr_tensorarrayunstack_1_tensorlistfromtensorcfoldr_while_tensorarrayv2read_1_tensorlistgetitem_foldr_tensorarrayunstack_1_tensorlistfromtensor_0"�
]foldr_while_tensorarrayv2read_tensorlistgetitem_foldr_tensorarrayunstack_tensorlistfromtensor_foldr_while_tensorarrayv2read_tensorlistgetitem_foldr_tensorarrayunstack_tensorlistfromtensor_0*$
_input_shapes
: : : :�: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:�:

_output_shapes
: :

_output_shapes
: 
�M
�
!__inference__traced_save_48585169
file_prefix'
#savev2_variable_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableopM
Isavev2_update_targets_periodic_update_targets_counter_read_readvariableop	6
2savev2_qnetwork_dense_2_kernel_read_readvariableop4
0savev2_qnetwork_dense_2_bias_read_readvariableopD
@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableopB
>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableopF
Bsavev2_qnetwork_encodingnetwork_dense_1_kernel_read_readvariableopD
@savev2_qnetwork_encodingnetwork_dense_1_bias_read_readvariableop<
8savev2_targetqnetwork_dense_5_kernel_read_readvariableop:
6savev2_targetqnetwork_dense_5_bias_read_readvariableopL
Hsavev2_targetqnetwork_encodingnetwork_dense_3_kernel_read_readvariableopJ
Fsavev2_targetqnetwork_encodingnetwork_dense_3_bias_read_readvariableopL
Hsavev2_targetqnetwork_encodingnetwork_dense_4_kernel_read_readvariableopJ
Fsavev2_targetqnetwork_encodingnetwork_dense_4_bias_read_readvariableopB
>savev2_rmsprop_qnetwork_dense_2_kernel_rms_read_readvariableop@
<savev2_rmsprop_qnetwork_dense_2_bias_rms_read_readvariableopP
Lsavev2_rmsprop_qnetwork_encodingnetwork_dense_kernel_rms_read_readvariableopN
Jsavev2_rmsprop_qnetwork_encodingnetwork_dense_bias_rms_read_readvariableopR
Nsavev2_rmsprop_qnetwork_encodingnetwork_dense_1_kernel_rms_read_readvariableopP
Lsavev2_rmsprop_qnetwork_encodingnetwork_dense_1_bias_rms_read_readvariableopA
=savev2_rmsprop_qnetwork_dense_2_kernel_mg_read_readvariableop?
;savev2_rmsprop_qnetwork_dense_2_bias_mg_read_readvariableopO
Ksavev2_rmsprop_qnetwork_encodingnetwork_dense_kernel_mg_read_readvariableopM
Isavev2_rmsprop_qnetwork_encodingnetwork_dense_bias_mg_read_readvariableopQ
Msavev2_rmsprop_qnetwork_encodingnetwork_dense_1_kernel_mg_read_readvariableopO
Ksavev2_rmsprop_qnetwork_encodingnetwork_dense_1_bias_mg_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B._train_step_counter/.ATTRIBUTES/VARIABLE_VALUEB*_optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+_optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB3_optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB._optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)_optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB2_update_target/_counter/.ATTRIBUTES/VARIABLE_VALUEB;_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB9_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB1_q_network/variables/0/.ATTRIBUTES/VARIABLE_VALUEB1_q_network/variables/1/.ATTRIBUTES/VARIABLE_VALUEB1_q_network/variables/2/.ATTRIBUTES/VARIABLE_VALUEB1_q_network/variables/3/.ATTRIBUTES/VARIABLE_VALUEBB_target_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB@_target_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB8_target_q_network/variables/0/.ATTRIBUTES/VARIABLE_VALUEB8_target_q_network/variables/1/.ATTRIBUTES/VARIABLE_VALUEB8_target_q_network/variables/2/.ATTRIBUTES/VARIABLE_VALUEB8_target_q_network/variables/3/.ATTRIBUTES/VARIABLE_VALUEBZ_q_network/_q_value_layer/kernel/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBX_q_network/_q_value_layer/bias/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBP_q_network/variables/0/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBP_q_network/variables/1/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBP_q_network/variables/2/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBP_q_network/variables/3/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBY_q_network/_q_value_layer/kernel/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBW_q_network/_q_value_layer/bias/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBO_q_network/variables/0/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBO_q_network/variables/1/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBO_q_network/variables/2/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBO_q_network/variables/3/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableopIsavev2_update_targets_periodic_update_targets_counter_read_readvariableop2savev2_qnetwork_dense_2_kernel_read_readvariableop0savev2_qnetwork_dense_2_bias_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableop>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableopBsavev2_qnetwork_encodingnetwork_dense_1_kernel_read_readvariableop@savev2_qnetwork_encodingnetwork_dense_1_bias_read_readvariableop8savev2_targetqnetwork_dense_5_kernel_read_readvariableop6savev2_targetqnetwork_dense_5_bias_read_readvariableopHsavev2_targetqnetwork_encodingnetwork_dense_3_kernel_read_readvariableopFsavev2_targetqnetwork_encodingnetwork_dense_3_bias_read_readvariableopHsavev2_targetqnetwork_encodingnetwork_dense_4_kernel_read_readvariableopFsavev2_targetqnetwork_encodingnetwork_dense_4_bias_read_readvariableop>savev2_rmsprop_qnetwork_dense_2_kernel_rms_read_readvariableop<savev2_rmsprop_qnetwork_dense_2_bias_rms_read_readvariableopLsavev2_rmsprop_qnetwork_encodingnetwork_dense_kernel_rms_read_readvariableopJsavev2_rmsprop_qnetwork_encodingnetwork_dense_bias_rms_read_readvariableopNsavev2_rmsprop_qnetwork_encodingnetwork_dense_1_kernel_rms_read_readvariableopLsavev2_rmsprop_qnetwork_encodingnetwork_dense_1_bias_rms_read_readvariableop=savev2_rmsprop_qnetwork_dense_2_kernel_mg_read_readvariableop;savev2_rmsprop_qnetwork_dense_2_bias_mg_read_readvariableopKsavev2_rmsprop_qnetwork_encodingnetwork_dense_kernel_mg_read_readvariableopIsavev2_rmsprop_qnetwork_encodingnetwork_dense_bias_mg_read_readvariableopMsavev2_rmsprop_qnetwork_encodingnetwork_dense_1_kernel_mg_read_readvariableopKsavev2_rmsprop_qnetwork_encodingnetwork_dense_1_bias_mg_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 		2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : : : :::	::::::	::::::	::::::	:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 	

_output_shapes
::$
 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
: 
�

�
foldr_while_cond_48584251(
$foldr_while_foldr_while_loop_counter.
*foldr_while_foldr_while_maximum_iterations
foldr_while_placeholder
foldr_while_placeholder_1B
>foldr_while_foldr_while_cond_48584251___redundant_placeholder0B
>foldr_while_foldr_while_cond_48584251___redundant_placeholder1
foldr_while_identity
p
foldr/while/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2
foldr/while/Greater/y�
foldr/while/GreaterGreaterfoldr_while_placeholderfoldr/while/Greater/y:output:0*
T0*
_output_shapes
: 2
foldr/while/Greater�
foldr/while/LessLess$foldr_while_foldr_while_loop_counter*foldr_while_foldr_while_maximum_iterations*
T0*
_output_shapes
: 2
foldr/while/Less�
foldr/while/LogicalAnd
LogicalAndfoldr/while/Less:z:0foldr/while/Greater:z:0*
_output_shapes
: 2
foldr/while/LogicalAndu
foldr/while/IdentityIdentityfoldr/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
foldr/while/Identity"5
foldr_while_identityfoldr/while/Identity:output:0*(
_input_shapes
: : : :�::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:�:

_output_shapes
::

_output_shapes
:
�!
�
cond_true_48585010 
cond_readvariableop_resource"
cond_assignvariableop_resource"
cond_readvariableop_1_resource$
 cond_assignvariableop_1_resource"
cond_readvariableop_2_resource$
 cond_assignvariableop_2_resource"
cond_readvariableop_3_resource$
 cond_assignvariableop_3_resource"
cond_readvariableop_4_resource$
 cond_assignvariableop_4_resource"
cond_readvariableop_5_resource$
 cond_assignvariableop_5_resource
cond_identity_equal

cond_identity_1
��cond/AssignVariableOp�cond/AssignVariableOp_1�cond/AssignVariableOp_2�cond/AssignVariableOp_3�cond/AssignVariableOp_4�cond/AssignVariableOp_5�cond/ReadVariableOp�cond/ReadVariableOp_1�cond/ReadVariableOp_2�cond/ReadVariableOp_3�cond/ReadVariableOp_4�cond/ReadVariableOp_5�
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes

:	*
dtype02
cond/ReadVariableOp�
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/ReadVariableOp:value:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp�
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_1_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_1�
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resourcecond/ReadVariableOp_1:value:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp_1�
cond/ReadVariableOp_2ReadVariableOpcond_readvariableop_2_resource*
_output_shapes

:*
dtype02
cond/ReadVariableOp_2�
cond/AssignVariableOp_2AssignVariableOp cond_assignvariableop_2_resourcecond/ReadVariableOp_2:value:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp_2�
cond/ReadVariableOp_3ReadVariableOpcond_readvariableop_3_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_3�
cond/AssignVariableOp_3AssignVariableOp cond_assignvariableop_3_resourcecond/ReadVariableOp_3:value:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp_3�
cond/ReadVariableOp_4ReadVariableOpcond_readvariableop_4_resource*
_output_shapes

:*
dtype02
cond/ReadVariableOp_4�
cond/AssignVariableOp_4AssignVariableOp cond_assignvariableop_4_resourcecond/ReadVariableOp_4:value:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp_4�
cond/ReadVariableOp_5ReadVariableOpcond_readvariableop_5_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_5�
cond/AssignVariableOp_5AssignVariableOp cond_assignvariableop_5_resourcecond/ReadVariableOp_5:value:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp_5�
cond/soft_variables_updateNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1^cond/AssignVariableOp_2^cond/AssignVariableOp_3^cond/AssignVariableOp_4^cond/AssignVariableOp_5",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
cond/soft_variables_update}
cond/IdentityIdentitycond_identity_equal^cond/soft_variables_update*
T0
*
_output_shapes
: 2
cond/Identity�
cond/Identity_1Identitycond/Identity:output:0^cond/AssignVariableOp^cond/AssignVariableOp_1^cond/AssignVariableOp_2^cond/AssignVariableOp_3^cond/AssignVariableOp_4^cond/AssignVariableOp_5^cond/ReadVariableOp^cond/ReadVariableOp_1^cond/ReadVariableOp_2^cond/ReadVariableOp_3^cond/ReadVariableOp_4^cond/ReadVariableOp_5*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*E
_input_shapes4
2::::::::::::: 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_122
cond/AssignVariableOp_2cond/AssignVariableOp_222
cond/AssignVariableOp_3cond/AssignVariableOp_322
cond/AssignVariableOp_4cond/AssignVariableOp_422
cond/AssignVariableOp_5cond/AssignVariableOp_52*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_12.
cond/ReadVariableOp_2cond/ReadVariableOp_22.
cond/ReadVariableOp_3cond/ReadVariableOp_32.
cond/ReadVariableOp_4cond/ReadVariableOp_42.
cond/ReadVariableOp_5cond/ReadVariableOp_5:

_output_shapes
: 
��
�
$__inference__traced_restore_48585272
file_prefix
assignvariableop_variable#
assignvariableop_1_rmsprop_iter$
 assignvariableop_2_rmsprop_decay,
(assignvariableop_3_rmsprop_learning_rate'
#assignvariableop_4_rmsprop_momentum"
assignvariableop_5_rmsprop_rhoE
Aassignvariableop_6_update_targets_periodic_update_targets_counter.
*assignvariableop_7_qnetwork_dense_2_kernel,
(assignvariableop_8_qnetwork_dense_2_bias<
8assignvariableop_9_qnetwork_encodingnetwork_dense_kernel;
7assignvariableop_10_qnetwork_encodingnetwork_dense_bias?
;assignvariableop_11_qnetwork_encodingnetwork_dense_1_kernel=
9assignvariableop_12_qnetwork_encodingnetwork_dense_1_bias5
1assignvariableop_13_targetqnetwork_dense_5_kernel3
/assignvariableop_14_targetqnetwork_dense_5_biasE
Aassignvariableop_15_targetqnetwork_encodingnetwork_dense_3_kernelC
?assignvariableop_16_targetqnetwork_encodingnetwork_dense_3_biasE
Aassignvariableop_17_targetqnetwork_encodingnetwork_dense_4_kernelC
?assignvariableop_18_targetqnetwork_encodingnetwork_dense_4_bias;
7assignvariableop_19_rmsprop_qnetwork_dense_2_kernel_rms9
5assignvariableop_20_rmsprop_qnetwork_dense_2_bias_rmsI
Eassignvariableop_21_rmsprop_qnetwork_encodingnetwork_dense_kernel_rmsG
Cassignvariableop_22_rmsprop_qnetwork_encodingnetwork_dense_bias_rmsK
Gassignvariableop_23_rmsprop_qnetwork_encodingnetwork_dense_1_kernel_rmsI
Eassignvariableop_24_rmsprop_qnetwork_encodingnetwork_dense_1_bias_rms:
6assignvariableop_25_rmsprop_qnetwork_dense_2_kernel_mg8
4assignvariableop_26_rmsprop_qnetwork_dense_2_bias_mgH
Dassignvariableop_27_rmsprop_qnetwork_encodingnetwork_dense_kernel_mgF
Bassignvariableop_28_rmsprop_qnetwork_encodingnetwork_dense_bias_mgJ
Fassignvariableop_29_rmsprop_qnetwork_encodingnetwork_dense_1_kernel_mgH
Dassignvariableop_30_rmsprop_qnetwork_encodingnetwork_dense_1_bias_mg
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B._train_step_counter/.ATTRIBUTES/VARIABLE_VALUEB*_optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+_optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB3_optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB._optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB)_optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB2_update_target/_counter/.ATTRIBUTES/VARIABLE_VALUEB;_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB9_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB1_q_network/variables/0/.ATTRIBUTES/VARIABLE_VALUEB1_q_network/variables/1/.ATTRIBUTES/VARIABLE_VALUEB1_q_network/variables/2/.ATTRIBUTES/VARIABLE_VALUEB1_q_network/variables/3/.ATTRIBUTES/VARIABLE_VALUEBB_target_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB@_target_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB8_target_q_network/variables/0/.ATTRIBUTES/VARIABLE_VALUEB8_target_q_network/variables/1/.ATTRIBUTES/VARIABLE_VALUEB8_target_q_network/variables/2/.ATTRIBUTES/VARIABLE_VALUEB8_target_q_network/variables/3/.ATTRIBUTES/VARIABLE_VALUEBZ_q_network/_q_value_layer/kernel/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBX_q_network/_q_value_layer/bias/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBP_q_network/variables/0/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBP_q_network/variables/1/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBP_q_network/variables/2/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBP_q_network/variables/3/.OPTIMIZER_SLOT/_optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBY_q_network/_q_value_layer/kernel/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBW_q_network/_q_value_layer/bias/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBO_q_network/variables/0/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBO_q_network/variables/1/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBO_q_network/variables/2/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEBO_q_network/variables/3/.OPTIMIZER_SLOT/_optimizer/mg/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_rmsprop_iterIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_rmsprop_decayIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp(assignvariableop_3_rmsprop_learning_rateIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_rmsprop_momentumIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_rmsprop_rhoIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpAassignvariableop_6_update_targets_periodic_update_targets_counterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp*assignvariableop_7_qnetwork_dense_2_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp(assignvariableop_8_qnetwork_dense_2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp8assignvariableop_9_qnetwork_encodingnetwork_dense_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_qnetwork_encodingnetwork_dense_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp;assignvariableop_11_qnetwork_encodingnetwork_dense_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp9assignvariableop_12_qnetwork_encodingnetwork_dense_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp1assignvariableop_13_targetqnetwork_dense_5_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp/assignvariableop_14_targetqnetwork_dense_5_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpAassignvariableop_15_targetqnetwork_encodingnetwork_dense_3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp?assignvariableop_16_targetqnetwork_encodingnetwork_dense_3_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpAassignvariableop_17_targetqnetwork_encodingnetwork_dense_4_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp?assignvariableop_18_targetqnetwork_encodingnetwork_dense_4_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp7assignvariableop_19_rmsprop_qnetwork_dense_2_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp5assignvariableop_20_rmsprop_qnetwork_dense_2_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpEassignvariableop_21_rmsprop_qnetwork_encodingnetwork_dense_kernel_rmsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpCassignvariableop_22_rmsprop_qnetwork_encodingnetwork_dense_bias_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpGassignvariableop_23_rmsprop_qnetwork_encodingnetwork_dense_1_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpEassignvariableop_24_rmsprop_qnetwork_encodingnetwork_dense_1_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp6assignvariableop_25_rmsprop_qnetwork_dense_2_kernel_mgIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp4assignvariableop_26_rmsprop_qnetwork_dense_2_bias_mgIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpDassignvariableop_27_rmsprop_qnetwork_encodingnetwork_dense_kernel_mgIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpBassignvariableop_28_rmsprop_qnetwork_encodingnetwork_dense_bias_mgIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpFassignvariableop_29_rmsprop_qnetwork_encodingnetwork_dense_1_kernel_mgIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOpDassignvariableop_30_rmsprop_qnetwork_encodingnetwork_dense_1_bias_mgIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31�
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*�
_input_shapes�
~: :::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
��
�1
__inference_train_48585055
experience_step_type
experience_observation
experience_action
experience_next_step_type
experience_reward
experience_discountF
Bloss_qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceG
Closs_qnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceH
Dloss_qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceI
Eloss_qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource8
4loss_qnetwork_dense_2_matmul_readvariableop_resource9
5loss_qnetwork_dense_2_biasadd_readvariableop_resourceN
Jloss_targetqnetwork_encodingnetwork_dense_3_matmul_readvariableop_resourceO
Kloss_targetqnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resourceN
Jloss_targetqnetwork_encodingnetwork_dense_4_matmul_readvariableop_resourceO
Kloss_targetqnetwork_encodingnetwork_dense_4_biasadd_readvariableop_resource>
:loss_targetqnetwork_dense_5_matmul_readvariableop_resource?
;loss_targetqnetwork_dense_5_biasadd_readvariableop_resource(
$rmsprop_cast_readvariableop_resource*
&rmsprop_cast_1_readvariableop_resource*
&rmsprop_cast_2_readvariableop_resource6
2rmsprop_rmsprop_update_mul_readvariableop_resource8
4rmsprop_rmsprop_update_mul_2_readvariableop_resource8
4rmsprop_rmsprop_update_1_mul_readvariableop_resource:
6rmsprop_rmsprop_update_1_mul_2_readvariableop_resource8
4rmsprop_rmsprop_update_2_mul_readvariableop_resource:
6rmsprop_rmsprop_update_2_mul_2_readvariableop_resource8
4rmsprop_rmsprop_update_3_mul_readvariableop_resource:
6rmsprop_rmsprop_update_3_mul_2_readvariableop_resource8
4rmsprop_rmsprop_update_4_mul_readvariableop_resource:
6rmsprop_rmsprop_update_4_mul_2_readvariableop_resource8
4rmsprop_rmsprop_update_5_mul_readvariableop_resource:
6rmsprop_rmsprop_update_5_mul_2_readvariableop_resource0
,rmsprop_rmsprop_assignaddvariableop_resource 
assignaddvariableop_resource"
assignaddvariableop_1_resource
identity

identity_1

identity_2��AssignAddVariableOp�AssignAddVariableOp_1�CheckNumerics�FloorMod/ReadVariableOp�RMSprop/Cast/ReadVariableOp�RMSprop/Cast_1/ReadVariableOp�RMSprop/Cast_2/ReadVariableOp�#RMSprop/RMSprop/AssignAddVariableOp�'RMSprop/RMSprop/update/AssignVariableOp�)RMSprop/RMSprop/update/AssignVariableOp_1�)RMSprop/RMSprop/update/AssignVariableOp_2�%RMSprop/RMSprop/update/ReadVariableOp�'RMSprop/RMSprop/update/ReadVariableOp_1�.RMSprop/RMSprop/update/Square_1/ReadVariableOp�)RMSprop/RMSprop/update/mul/ReadVariableOp�+RMSprop/RMSprop/update/mul_2/ReadVariableOp�)RMSprop/RMSprop/update_1/AssignVariableOp�+RMSprop/RMSprop/update_1/AssignVariableOp_1�+RMSprop/RMSprop/update_1/AssignVariableOp_2�'RMSprop/RMSprop/update_1/ReadVariableOp�)RMSprop/RMSprop/update_1/ReadVariableOp_1�0RMSprop/RMSprop/update_1/Square_1/ReadVariableOp�+RMSprop/RMSprop/update_1/mul/ReadVariableOp�-RMSprop/RMSprop/update_1/mul_2/ReadVariableOp�)RMSprop/RMSprop/update_2/AssignVariableOp�+RMSprop/RMSprop/update_2/AssignVariableOp_1�+RMSprop/RMSprop/update_2/AssignVariableOp_2�'RMSprop/RMSprop/update_2/ReadVariableOp�)RMSprop/RMSprop/update_2/ReadVariableOp_1�0RMSprop/RMSprop/update_2/Square_1/ReadVariableOp�+RMSprop/RMSprop/update_2/mul/ReadVariableOp�-RMSprop/RMSprop/update_2/mul_2/ReadVariableOp�)RMSprop/RMSprop/update_3/AssignVariableOp�+RMSprop/RMSprop/update_3/AssignVariableOp_1�+RMSprop/RMSprop/update_3/AssignVariableOp_2�'RMSprop/RMSprop/update_3/ReadVariableOp�)RMSprop/RMSprop/update_3/ReadVariableOp_1�0RMSprop/RMSprop/update_3/Square_1/ReadVariableOp�+RMSprop/RMSprop/update_3/mul/ReadVariableOp�-RMSprop/RMSprop/update_3/mul_2/ReadVariableOp�)RMSprop/RMSprop/update_4/AssignVariableOp�+RMSprop/RMSprop/update_4/AssignVariableOp_1�+RMSprop/RMSprop/update_4/AssignVariableOp_2�'RMSprop/RMSprop/update_4/ReadVariableOp�)RMSprop/RMSprop/update_4/ReadVariableOp_1�0RMSprop/RMSprop/update_4/Square_1/ReadVariableOp�+RMSprop/RMSprop/update_4/mul/ReadVariableOp�-RMSprop/RMSprop/update_4/mul_2/ReadVariableOp�)RMSprop/RMSprop/update_5/AssignVariableOp�+RMSprop/RMSprop/update_5/AssignVariableOp_1�+RMSprop/RMSprop/update_5/AssignVariableOp_2�'RMSprop/RMSprop/update_5/ReadVariableOp�)RMSprop/RMSprop/update_5/ReadVariableOp_1�0RMSprop/RMSprop/update_5/Square_1/ReadVariableOp�+RMSprop/RMSprop/update_5/mul/ReadVariableOp�-RMSprop/RMSprop/update_5/mul_2/ReadVariableOp�%Variables/StopGradient/ReadVariableOp�'Variables/StopGradient_1/ReadVariableOp�'Variables/StopGradient_2/ReadVariableOp�'Variables/StopGradient_3/ReadVariableOp�'Variables/StopGradient_4/ReadVariableOp�'Variables/StopGradient_5/ReadVariableOp�cond�:loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�9loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�<loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�;loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�,loss/QNetwork/dense_2/BiasAdd/ReadVariableOp�+loss/QNetwork/dense_2/MatMul/ReadVariableOp�Bloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp�Dloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOp�Aloss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp�Closs/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOp�Bloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOp�Dloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOp�Aloss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOp�Closs/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOp�2loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOp�4loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOp�1loss/TargetQNetwork/dense_5/MatMul/ReadVariableOp�3loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOp�*summarize_vars/StopGradient/ReadVariableOp�,summarize_vars/StopGradient_1/ReadVariableOp�,summarize_vars/StopGradient_2/ReadVariableOp�,summarize_vars/StopGradient_3/ReadVariableOp�,summarize_vars/StopGradient_4/ReadVariableOp�,summarize_vars/StopGradient_5/ReadVariableOp�)summarize_vars/global_norm/ReadVariableOp�+summarize_vars/global_norm_1/ReadVariableOp�+summarize_vars/global_norm_2/ReadVariableOp�+summarize_vars/global_norm_3/ReadVariableOp�+summarize_vars/global_norm_4/ReadVariableOp�+summarize_vars/global_norm_5/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceexperience_step_typestrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slicer
SqueezeSqueezestrided_slice:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2	
Squeeze�
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"               2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_1/stack_2�
strided_slice_1StridedSliceexperience_observationstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�	*

begin_mask*
end_mask2
strided_slice_1�
	Squeeze_1Squeezestrided_slice_1:output:0*
T0*#
_output_shapes
:�	*
squeeze_dims
2
	Squeeze_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceexperience_actionstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_2x
	Squeeze_2Squeezestrided_slice_2:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2
	Squeeze_2
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2�
strided_slice_3StridedSliceexperience_next_step_typestrided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_3x
	Squeeze_3Squeezestrided_slice_3:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2
	Squeeze_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2�
strided_slice_4StridedSliceexperience_rewardstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_4x
	Squeeze_4Squeezestrided_slice_4:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2
	Squeeze_4
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack�
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_5/stack_1�
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2�
strided_slice_5StridedSliceexperience_discountstrided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_5x
	Squeeze_5Squeezestrided_slice_5:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2
	Squeeze_5
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_6/stack�
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack_1�
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2�
strided_slice_6StridedSliceexperience_step_typestrided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_6x
	Squeeze_6Squeezestrided_slice_6:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2
	Squeeze_6�
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*%
valueB"    ����        2
strided_slice_7/stack�
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_7/stack_1�
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_7/stack_2�
strided_slice_7StridedSliceexperience_observationstrided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*'
_output_shapes
:�	*

begin_mask*
end_mask2
strided_slice_7�
	Squeeze_7Squeezestrided_slice_7:output:0*
T0*#
_output_shapes
:�	*
squeeze_dims
2
	Squeeze_7
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_8/stack�
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_8/stack_1�
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_8/stack_2�
strided_slice_8StridedSliceexperience_actionstrided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_8x
	Squeeze_8Squeezestrided_slice_8:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2
	Squeeze_8
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_9/stack�
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_9/stack_1�
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_9/stack_2�
strided_slice_9StridedSliceexperience_next_step_typestrided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_9x
	Squeeze_9Squeezestrided_slice_9:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2
	Squeeze_9�
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_10/stack�
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1�
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2�
strided_slice_10StridedSliceexperience_rewardstrided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_10{

Squeeze_10Squeezestrided_slice_10:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2

Squeeze_10�
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_11/stack�
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_11/stack_1�
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_11/stack_2�
strided_slice_11StridedSliceexperience_discountstrided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_11{

Squeeze_11Squeezestrided_slice_11:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
2

Squeeze_11�
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_12/stack�
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_12/stack_1�
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_12/stack_2�
strided_slice_12StridedSliceexperience_rewardstrided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_12�
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_13/stack�
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ����2
strided_slice_13/stack_1�
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_13/stack_2�
strided_slice_13StridedSliceexperience_discountstrided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*
_output_shapes
:	�*

begin_mask*
end_mask2
strided_slice_13S
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
mul/xf
mulMulmul/x:output:0strided_slice_13:output:0*
T0*
_output_shapes
:	�2
mul�
$to_time_major_tensors/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2&
$to_time_major_tensors/transpose/perm�
to_time_major_tensors/transpose	Transposemul:z:0-to_time_major_tensors/transpose/perm:output:0*
T0*
_output_shapes
:	�2!
to_time_major_tensors/transpose�
&to_time_major_tensors/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2(
&to_time_major_tensors/transpose_1/perm�
!to_time_major_tensors/transpose_1	Transposestrided_slice_12:output:0/to_time_major_tensors/transpose_1/perm:output:0*
T0*
_output_shapes
:	�2#
!to_time_major_tensors/transpose_1�
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������2
strided_slice_14/stack~
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_14/stack_1~
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_2�
strided_slice_14StridedSlice%to_time_major_tensors/transpose_1:y:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes	
:�*
shrink_axis_mask2
strided_slice_14g

zeros_likeConst*
_output_shapes	
:�*
dtype0*
valueB�*    2

zeros_like�
!foldr/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2#
!foldr/TensorArrayV2/element_shape�
 foldr/TensorArrayV2/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :2"
 foldr/TensorArrayV2/num_elements�
foldr/TensorArrayV2TensorListReserve*foldr/TensorArrayV2/element_shape:output:0)foldr/TensorArrayV2/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
foldr/TensorArrayV2�
;foldr/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:�2=
;foldr/TensorArrayUnstack/TensorListFromTensor/element_shape�
-foldr/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%to_time_major_tensors/transpose_1:y:0Dfoldr/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02/
-foldr/TensorArrayUnstack/TensorListFromTensor�
#foldr/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#foldr/TensorArrayV2_1/element_shape�
"foldr/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :2$
"foldr/TensorArrayV2_1/num_elements�
foldr/TensorArrayV2_1TensorListReserve,foldr/TensorArrayV2_1/element_shape:output:0+foldr/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
foldr/TensorArrayV2_1�
=foldr/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:�2?
=foldr/TensorArrayUnstack_1/TensorListFromTensor/element_shape�
/foldr/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor#to_time_major_tensors/transpose:y:0Ffoldr/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/foldr/TensorArrayUnstack_1/TensorListFromTensor\
foldr/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
foldr/Const�
foldr/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
value	B :2 
foldr/while/maximum_iterationsv
foldr/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
foldr/while/loop_counter�
foldr/whileStatelessWhile!foldr/while/loop_counter:output:0'foldr/while/maximum_iterations:output:0foldr/Const:output:0zeros_like:output:0=foldr/TensorArrayUnstack/TensorListFromTensor:output_handle:0?foldr/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*%
_output_shapes
: : : :�: : * 
_read_only_resource_inputs
 *%
bodyR
foldr_while_body_48584252*%
condR
foldr_while_cond_48584251*$
output_shapes
: : : :�: : 2
foldr/whileo
foldr/StopGradientStopGradientfoldr/while:output:2*
T0*
_output_shapes
: 2
foldr/StopGradientx
foldr/StopGradient_1StopGradientfoldr/while:output:3*
T0*
_output_shapes	
:�2
foldr/StopGradient_1q
StopGradientStopGradientfoldr/StopGradient_1:output:0*
T0*
_output_shapes	
:�2
StopGradientr
Prod/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Prod/reduction_indicesv
ProdProdstrided_slice_13:output:0Prod/reduction_indices:output:0*
T0*
_output_shapes	
:�2
ProdW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
mul_1/x\
mul_1Mulmul_1/x:output:0Prod:output:0*
T0*
_output_shapes	
:�2
mul_1m
ones_like/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ones_like/Constx
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*
_output_shapes	
:�2
	ones_likeW
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �2	
mul_2/xa
mul_2Mulmul_2/x:output:0ones_like:output:0*
T0*
_output_shapes	
:�2
mul_2q
ones_like_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ones_like_1/Const�
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*
_output_shapes	
:�2
ones_like_1W
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �2	
mul_3/xc
mul_3Mulmul_3/x:output:0ones_like_1:output:0*
T0*
_output_shapes	
:�2
mul_3�
"loss/QNetwork/EncodingNetwork/CastCastSqueeze_1:output:0*

DstT0*

SrcT0*#
_output_shapes
:�	2$
"loss/QNetwork/EncodingNetwork/Cast�
+loss/QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����	   2-
+loss/QNetwork/EncodingNetwork/flatten/Const�
-loss/QNetwork/EncodingNetwork/flatten/ReshapeReshape&loss/QNetwork/EncodingNetwork/Cast:y:04loss/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*
_output_shapes
:	�	2/
-loss/QNetwork/EncodingNetwork/flatten/Reshape�
9loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpBloss_qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02;
9loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
*loss/QNetwork/EncodingNetwork/dense/MatMulMatMul6loss/QNetwork/EncodingNetwork/flatten/Reshape:output:0Aloss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2,
*loss/QNetwork/EncodingNetwork/dense/MatMul�
:loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpCloss_qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
+loss/QNetwork/EncodingNetwork/dense/BiasAddBiasAdd4loss/QNetwork/EncodingNetwork/dense/MatMul:product:0Bloss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2-
+loss/QNetwork/EncodingNetwork/dense/BiasAdd�
+loss/QNetwork/EncodingNetwork/dense/SigmoidSigmoid4loss/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*
_output_shapes
:	�2-
+loss/QNetwork/EncodingNetwork/dense/Sigmoid�
'loss/QNetwork/EncodingNetwork/dense/mulMul4loss/QNetwork/EncodingNetwork/dense/BiasAdd:output:0/loss/QNetwork/EncodingNetwork/dense/Sigmoid:y:0*
T0*
_output_shapes
:	�2)
'loss/QNetwork/EncodingNetwork/dense/mul�
,loss/QNetwork/EncodingNetwork/dense/IdentityIdentity+loss/QNetwork/EncodingNetwork/dense/mul:z:0*
T0*
_output_shapes
:	�2.
,loss/QNetwork/EncodingNetwork/dense/Identity�
-loss/QNetwork/EncodingNetwork/dense/IdentityN	IdentityN+loss/QNetwork/EncodingNetwork/dense/mul:z:04loss/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-48584320**
_output_shapes
:	�:	�2/
-loss/QNetwork/EncodingNetwork/dense/IdentityN�
;loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpDloss_qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02=
;loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�
,loss/QNetwork/EncodingNetwork/dense_1/MatMulMatMul6loss/QNetwork/EncodingNetwork/dense/IdentityN:output:0Closs/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2.
,loss/QNetwork/EncodingNetwork/dense_1/MatMul�
<loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpEloss_qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�
-loss/QNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd6loss/QNetwork/EncodingNetwork/dense_1/MatMul:product:0Dloss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2/
-loss/QNetwork/EncodingNetwork/dense_1/BiasAdd�
-loss/QNetwork/EncodingNetwork/dense_1/SigmoidSigmoid6loss/QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	�2/
-loss/QNetwork/EncodingNetwork/dense_1/Sigmoid�
)loss/QNetwork/EncodingNetwork/dense_1/mulMul6loss/QNetwork/EncodingNetwork/dense_1/BiasAdd:output:01loss/QNetwork/EncodingNetwork/dense_1/Sigmoid:y:0*
T0*
_output_shapes
:	�2+
)loss/QNetwork/EncodingNetwork/dense_1/mul�
.loss/QNetwork/EncodingNetwork/dense_1/IdentityIdentity-loss/QNetwork/EncodingNetwork/dense_1/mul:z:0*
T0*
_output_shapes
:	�20
.loss/QNetwork/EncodingNetwork/dense_1/Identity�
/loss/QNetwork/EncodingNetwork/dense_1/IdentityN	IdentityN-loss/QNetwork/EncodingNetwork/dense_1/mul:z:06loss/QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-48584332**
_output_shapes
:	�:	�21
/loss/QNetwork/EncodingNetwork/dense_1/IdentityN�
+loss/QNetwork/dense_2/MatMul/ReadVariableOpReadVariableOp4loss_qnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+loss/QNetwork/dense_2/MatMul/ReadVariableOp�
loss/QNetwork/dense_2/MatMulMatMul8loss/QNetwork/EncodingNetwork/dense_1/IdentityN:output:03loss/QNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
loss/QNetwork/dense_2/MatMul�
,loss/QNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp5loss_qnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,loss/QNetwork/dense_2/BiasAdd/ReadVariableOp�
loss/QNetwork/dense_2/BiasAddBiasAdd&loss/QNetwork/dense_2/MatMul:product:04loss/QNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2
loss/QNetwork/dense_2/BiasAddc

loss/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�2

loss/Shape~
loss/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
loss/strided_slice/stack�
loss/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
loss/strided_slice/stack_1�
loss/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
loss/strided_slice/stack_2�
loss/strided_sliceStridedSliceloss/Shape:output:0!loss/strided_slice/stack:output:0#loss/strided_slice/stack_1:output:0#loss/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
loss/strided_slicef
loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
loss/range/startf
loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
loss/range/delta�

loss/rangeRangeloss/range/start:output:0loss/strided_slice:output:0loss/range/delta:output:0*
_output_shapes	
:�2

loss/range�
loss/meshgrid/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2
loss/meshgrid/Reshape/shape�
loss/meshgrid/ReshapeReshapeloss/range:output:0$loss/meshgrid/Reshape/shape:output:0*
T0*
_output_shapes	
:�2
loss/meshgrid/Reshapek
loss/meshgrid/SizeConst*
_output_shapes
: *
dtype0*
value
B :�2
loss/meshgrid/Sizey
loss/meshgrid/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
loss/meshgrid/ones/Less/y�
loss/meshgrid/ones/LessLessloss/meshgrid/Size:output:0"loss/meshgrid/ones/Less/y:output:0*
T0*
_output_shapes
: 2
loss/meshgrid/ones/Less�
loss/meshgrid/ones/packedPackloss/meshgrid/Size:output:0*
N*
T0*
_output_shapes
:2
loss/meshgrid/ones/packedv
loss/meshgrid/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
loss/meshgrid/ones/Const�
loss/meshgrid/onesFill"loss/meshgrid/ones/packed:output:0!loss/meshgrid/ones/Const:output:0*
T0*
_output_shapes	
:�2
loss/meshgrid/ones�
loss/meshgrid/mulMulloss/meshgrid/Reshape:output:0loss/meshgrid/ones:output:0*
T0*
_output_shapes	
:�2
loss/meshgrid/mulu
loss/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
loss/ExpandDims/dim�
loss/ExpandDims
ExpandDimsloss/meshgrid/mul:z:0loss/ExpandDims/dim:output:0*
T0*
_output_shapes
:	�2
loss/ExpandDimsy
loss/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
loss/ExpandDims_1/dim�
loss/ExpandDims_1
ExpandDimsSqueeze_2:output:0loss/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:	�2
loss/ExpandDims_1o
loss/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
loss/concat/axis�
loss/concatConcatV2loss/ExpandDims:output:0loss/ExpandDims_1:output:0loss/concat/axis:output:0*
N*
T0*
_output_shapes
:	�2
loss/concat�
loss/GatherNdGatherNd&loss/QNetwork/dense_2/BiasAdd:output:0loss/concat:output:0*
Tindices0*
Tparams0*
_output_shapes	
:�2
loss/GatherNd�
(loss/TargetQNetwork/EncodingNetwork/CastCastSqueeze_7:output:0*

DstT0*

SrcT0*#
_output_shapes
:�	2*
(loss/TargetQNetwork/EncodingNetwork/Cast�
3loss/TargetQNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����	   25
3loss/TargetQNetwork/EncodingNetwork/flatten_1/Const�
5loss/TargetQNetwork/EncodingNetwork/flatten_1/ReshapeReshape,loss/TargetQNetwork/EncodingNetwork/Cast:y:0<loss/TargetQNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*
_output_shapes
:	�	27
5loss/TargetQNetwork/EncodingNetwork/flatten_1/Reshape�
Aloss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpReadVariableOpJloss_targetqnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02C
Aloss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp�
2loss/TargetQNetwork/EncodingNetwork/dense_3/MatMulMatMul>loss/TargetQNetwork/EncodingNetwork/flatten_1/Reshape:output:0Iloss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�24
2loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul�
Bloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOpKloss_targetqnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp�
3loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAddBiasAdd<loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul:product:0Jloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd�
3loss/TargetQNetwork/EncodingNetwork/dense_3/SigmoidSigmoid<loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:	�25
3loss/TargetQNetwork/EncodingNetwork/dense_3/Sigmoid�
/loss/TargetQNetwork/EncodingNetwork/dense_3/mulMul<loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd:output:07loss/TargetQNetwork/EncodingNetwork/dense_3/Sigmoid:y:0*
T0*
_output_shapes
:	�21
/loss/TargetQNetwork/EncodingNetwork/dense_3/mul�
4loss/TargetQNetwork/EncodingNetwork/dense_3/IdentityIdentity3loss/TargetQNetwork/EncodingNetwork/dense_3/mul:z:0*
T0*
_output_shapes
:	�26
4loss/TargetQNetwork/EncodingNetwork/dense_3/Identity�
5loss/TargetQNetwork/EncodingNetwork/dense_3/IdentityN	IdentityN3loss/TargetQNetwork/EncodingNetwork/dense_3/mul:z:0<loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-48584377**
_output_shapes
:	�:	�27
5loss/TargetQNetwork/EncodingNetwork/dense_3/IdentityN�
Aloss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOpReadVariableOpJloss_targetqnetwork_encodingnetwork_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02C
Aloss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOp�
2loss/TargetQNetwork/EncodingNetwork/dense_4/MatMulMatMul>loss/TargetQNetwork/EncodingNetwork/dense_3/IdentityN:output:0Iloss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�24
2loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul�
Bloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOpReadVariableOpKloss_targetqnetwork_encodingnetwork_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOp�
3loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAddBiasAdd<loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul:product:0Jloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd�
3loss/TargetQNetwork/EncodingNetwork/dense_4/SigmoidSigmoid<loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd:output:0*
T0*
_output_shapes
:	�25
3loss/TargetQNetwork/EncodingNetwork/dense_4/Sigmoid�
/loss/TargetQNetwork/EncodingNetwork/dense_4/mulMul<loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd:output:07loss/TargetQNetwork/EncodingNetwork/dense_4/Sigmoid:y:0*
T0*
_output_shapes
:	�21
/loss/TargetQNetwork/EncodingNetwork/dense_4/mul�
4loss/TargetQNetwork/EncodingNetwork/dense_4/IdentityIdentity3loss/TargetQNetwork/EncodingNetwork/dense_4/mul:z:0*
T0*
_output_shapes
:	�26
4loss/TargetQNetwork/EncodingNetwork/dense_4/Identity�
5loss/TargetQNetwork/EncodingNetwork/dense_4/IdentityN	IdentityN3loss/TargetQNetwork/EncodingNetwork/dense_4/mul:z:0<loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd:output:0*
T
2*.
_gradient_op_typeCustomGradient-48584389**
_output_shapes
:	�:	�27
5loss/TargetQNetwork/EncodingNetwork/dense_4/IdentityN�
1loss/TargetQNetwork/dense_5/MatMul/ReadVariableOpReadVariableOp:loss_targetqnetwork_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype023
1loss/TargetQNetwork/dense_5/MatMul/ReadVariableOp�
"loss/TargetQNetwork/dense_5/MatMulMatMul>loss/TargetQNetwork/EncodingNetwork/dense_4/IdentityN:output:09loss/TargetQNetwork/dense_5/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2$
"loss/TargetQNetwork/dense_5/MatMul�
2loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOpReadVariableOp;loss_targetqnetwork_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOp�
#loss/TargetQNetwork/dense_5/BiasAddBiasAdd,loss/TargetQNetwork/dense_5/MatMul:product:0:loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2%
#loss/TargetQNetwork/dense_5/BiasAdd�
*loss/TargetQNetwork/EncodingNetwork/Cast_1CastSqueeze_7:output:0*

DstT0*

SrcT0*#
_output_shapes
:�	2,
*loss/TargetQNetwork/EncodingNetwork/Cast_1�
5loss/TargetQNetwork/EncodingNetwork/flatten_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"����	   27
5loss/TargetQNetwork/EncodingNetwork/flatten_1/Const_1�
7loss/TargetQNetwork/EncodingNetwork/flatten_1/Reshape_1Reshape.loss/TargetQNetwork/EncodingNetwork/Cast_1:y:0>loss/TargetQNetwork/EncodingNetwork/flatten_1/Const_1:output:0*
T0*
_output_shapes
:	�	29
7loss/TargetQNetwork/EncodingNetwork/flatten_1/Reshape_1�
Closs/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOpReadVariableOpJloss_targetqnetwork_encodingnetwork_dense_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02E
Closs/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOp�
4loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1MatMul@loss/TargetQNetwork/EncodingNetwork/flatten_1/Reshape_1:output:0Kloss/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�26
4loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1�
Dloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOpReadVariableOpKloss_targetqnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02F
Dloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOp�
5loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1BiasAdd>loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1:product:0Lloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�27
5loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1�
5loss/TargetQNetwork/EncodingNetwork/dense_3/Sigmoid_1Sigmoid>loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1:output:0*
T0*
_output_shapes
:	�27
5loss/TargetQNetwork/EncodingNetwork/dense_3/Sigmoid_1�
1loss/TargetQNetwork/EncodingNetwork/dense_3/mul_1Mul>loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1:output:09loss/TargetQNetwork/EncodingNetwork/dense_3/Sigmoid_1:y:0*
T0*
_output_shapes
:	�23
1loss/TargetQNetwork/EncodingNetwork/dense_3/mul_1�
6loss/TargetQNetwork/EncodingNetwork/dense_3/Identity_1Identity5loss/TargetQNetwork/EncodingNetwork/dense_3/mul_1:z:0*
T0*
_output_shapes
:	�28
6loss/TargetQNetwork/EncodingNetwork/dense_3/Identity_1�
7loss/TargetQNetwork/EncodingNetwork/dense_3/IdentityN_1	IdentityN5loss/TargetQNetwork/EncodingNetwork/dense_3/mul_1:z:0>loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1:output:0*
T
2*.
_gradient_op_typeCustomGradient-48584408**
_output_shapes
:	�:	�29
7loss/TargetQNetwork/EncodingNetwork/dense_3/IdentityN_1�
Closs/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOpReadVariableOpJloss_targetqnetwork_encodingnetwork_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02E
Closs/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOp�
4loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1MatMul@loss/TargetQNetwork/EncodingNetwork/dense_3/IdentityN_1:output:0Kloss/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�26
4loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1�
Dloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOpReadVariableOpKloss_targetqnetwork_encodingnetwork_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02F
Dloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOp�
5loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1BiasAdd>loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1:product:0Lloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�27
5loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1�
5loss/TargetQNetwork/EncodingNetwork/dense_4/Sigmoid_1Sigmoid>loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1:output:0*
T0*
_output_shapes
:	�27
5loss/TargetQNetwork/EncodingNetwork/dense_4/Sigmoid_1�
1loss/TargetQNetwork/EncodingNetwork/dense_4/mul_1Mul>loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1:output:09loss/TargetQNetwork/EncodingNetwork/dense_4/Sigmoid_1:y:0*
T0*
_output_shapes
:	�23
1loss/TargetQNetwork/EncodingNetwork/dense_4/mul_1�
6loss/TargetQNetwork/EncodingNetwork/dense_4/Identity_1Identity5loss/TargetQNetwork/EncodingNetwork/dense_4/mul_1:z:0*
T0*
_output_shapes
:	�28
6loss/TargetQNetwork/EncodingNetwork/dense_4/Identity_1�
7loss/TargetQNetwork/EncodingNetwork/dense_4/IdentityN_1	IdentityN5loss/TargetQNetwork/EncodingNetwork/dense_4/mul_1:z:0>loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1:output:0*
T
2*.
_gradient_op_typeCustomGradient-48584418**
_output_shapes
:	�:	�29
7loss/TargetQNetwork/EncodingNetwork/dense_4/IdentityN_1�
3loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOpReadVariableOp:loss_targetqnetwork_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype025
3loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOp�
$loss/TargetQNetwork/dense_5/MatMul_1MatMul@loss/TargetQNetwork/EncodingNetwork/dense_4/IdentityN_1:output:0;loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2&
$loss/TargetQNetwork/dense_5/MatMul_1�
4loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOpReadVariableOp;loss_targetqnetwork_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOp�
%loss/TargetQNetwork/dense_5/BiasAdd_1BiasAdd.loss/TargetQNetwork/dense_5/MatMul_1:product:0<loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	�2'
%loss/TargetQNetwork/dense_5/BiasAdd_1�
+loss/loss_Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2-
+loss/loss_Categorical/mode/ArgMax/dimension�
!loss/loss_Categorical/mode/ArgMaxArgMax.loss/TargetQNetwork/dense_5/BiasAdd_1:output:04loss/loss_Categorical/mode/ArgMax/dimension:output:0*
T0*
_output_shapes	
:�2#
!loss/loss_Categorical/mode/ArgMax�
loss/loss_Categorical/mode/CastCast*loss/loss_Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*
_output_shapes	
:�2!
loss/loss_Categorical/mode/Castt
loss/Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
loss/Deterministic/atolt
loss/Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
loss/Deterministic/rtol�
+loss/loss_Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2-
+loss/loss_Deterministic/sample/sample_shape�
$loss/loss_Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2&
$loss/loss_Deterministic/sample/Const�
0loss/loss_Deterministic/sample/BroadcastTo/shapeConst*
_output_shapes
:*
dtype0*
valueB"   �   22
0loss/loss_Deterministic/sample/BroadcastTo/shape�
*loss/loss_Deterministic/sample/BroadcastToBroadcastTo#loss/loss_Categorical/mode/Cast:y:09loss/loss_Deterministic/sample/BroadcastTo/shape:output:0*
T0*
_output_shapes
:	�2,
*loss/loss_Deterministic/sample/BroadcastTo�
,loss/loss_Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:�2.
,loss/loss_Deterministic/sample/Reshape/shape�
&loss/loss_Deterministic/sample/ReshapeReshape3loss/loss_Deterministic/sample/BroadcastTo:output:05loss/loss_Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes	
:�2(
&loss/loss_Deterministic/sample/Reshape~
loss/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
loss/clip_by_value/Minimum/y�
loss/clip_by_value/MinimumMinimum/loss/loss_Deterministic/sample/Reshape:output:0%loss/clip_by_value/Minimum/y:output:0*
T0*
_output_shapes	
:�2
loss/clip_by_value/Minimumn
loss/clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
loss/clip_by_value/y�
loss/clip_by_valueMaximumloss/clip_by_value/Minimum:z:0loss/clip_by_value/y:output:0*
T0*
_output_shapes	
:�2
loss/clip_by_valueg
loss/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�2
loss/Shape_1�
loss/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
loss/strided_slice_1/stack�
loss/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
loss/strided_slice_1/stack_1�
loss/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
loss/strided_slice_1/stack_2�
loss/strided_slice_1StridedSliceloss/Shape_1:output:0#loss/strided_slice_1/stack:output:0%loss/strided_slice_1/stack_1:output:0%loss/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
loss/strided_slice_1j
loss/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
loss/range_1/startj
loss/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
loss/range_1/delta�
loss/range_1Rangeloss/range_1/start:output:0loss/strided_slice_1:output:0loss/range_1/delta:output:0*
_output_shapes	
:�2
loss/range_1�
loss/meshgrid_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
���������2
loss/meshgrid_1/Reshape/shape�
loss/meshgrid_1/ReshapeReshapeloss/range_1:output:0&loss/meshgrid_1/Reshape/shape:output:0*
T0*
_output_shapes	
:�2
loss/meshgrid_1/Reshapeo
loss/meshgrid_1/SizeConst*
_output_shapes
: *
dtype0*
value
B :�2
loss/meshgrid_1/Size}
loss/meshgrid_1/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :�2
loss/meshgrid_1/ones/Less/y�
loss/meshgrid_1/ones/LessLessloss/meshgrid_1/Size:output:0$loss/meshgrid_1/ones/Less/y:output:0*
T0*
_output_shapes
: 2
loss/meshgrid_1/ones/Less�
loss/meshgrid_1/ones/packedPackloss/meshgrid_1/Size:output:0*
N*
T0*
_output_shapes
:2
loss/meshgrid_1/ones/packedz
loss/meshgrid_1/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
loss/meshgrid_1/ones/Const�
loss/meshgrid_1/onesFill$loss/meshgrid_1/ones/packed:output:0#loss/meshgrid_1/ones/Const:output:0*
T0*
_output_shapes	
:�2
loss/meshgrid_1/ones�
loss/meshgrid_1/mulMul loss/meshgrid_1/Reshape:output:0loss/meshgrid_1/ones:output:0*
T0*
_output_shapes	
:�2
loss/meshgrid_1/muly
loss/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
loss/ExpandDims_2/dim�
loss/ExpandDims_2
ExpandDimsloss/meshgrid_1/mul:z:0loss/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:	�2
loss/ExpandDims_2y
loss/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
valueB :
���������2
loss/ExpandDims_3/dim�
loss/ExpandDims_3
ExpandDimsloss/clip_by_value:z:0loss/ExpandDims_3/dim:output:0*
T0*
_output_shapes
:	�2
loss/ExpandDims_3s
loss/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
loss/concat_1/axis�
loss/concat_1ConcatV2loss/ExpandDims_2:output:0loss/ExpandDims_3:output:0loss/concat_1/axis:output:0*
N*
T0*
_output_shapes
:	�2
loss/concat_1�
loss/GatherNd_1GatherNd,loss/TargetQNetwork/dense_5/BiasAdd:output:0loss/concat_1:output:0*
Tindices0*
Tparams0*
_output_shapes	
:�2
loss/GatherNd_1]

loss/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

loss/mul/xm
loss/mulMulloss/mul/x:output:0StopGradient:output:0*
T0*
_output_shapes	
:�2

loss/mula
loss/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
loss/mul_1/xg

loss/mul_1Mulloss/mul_1/x:output:0	mul_1:z:0*
T0*
_output_shapes	
:�2

loss/mul_1o

loss/mul_2Mulloss/mul_1:z:0loss/GatherNd_1:output:0*
T0*
_output_shapes	
:�2

loss/mul_2a
loss/addAddV2loss/mul:z:0loss/mul_2:z:0*
T0*
_output_shapes	
:�2

loss/addj
loss/StopGradientStopGradientloss/add:z:0*
T0*
_output_shapes	
:�2
loss/StopGradient^
loss/Equal/yConst*
_output_shapes
: *
dtype0*
value	B :2
loss/Equal/yp

loss/EqualEqualSqueeze:output:0loss/Equal/y:output:0*
T0*
_output_shapes	
:�2

loss/Equal]
loss/LogicalNot
LogicalNotloss/Equal:z:0*
_output_shapes	
:�2
loss/LogicalNoth
	loss/CastCastloss/LogicalNot:y:0*

DstT0*

SrcT0
*
_output_shapes	
:�2
	loss/Castu
loss/subSubloss/StopGradient:output:0loss/GatherNd:output:0*
T0*
_output_shapes	
:�2

loss/subb

loss/mul_3Mulloss/Cast:y:0loss/sub:z:0*
T0*
_output_shapes	
:�2

loss/mul_3u
loss/huber_loss/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
loss/huber_loss/Cast/x�
loss/huber_loss/SubSubloss/GatherNd:output:0loss/StopGradient:output:0*
T0*
_output_shapes	
:�2
loss/huber_loss/Subp
loss/huber_loss/AbsAbsloss/huber_loss/Sub:z:0*
T0*
_output_shapes	
:�2
loss/huber_loss/Abss
loss/huber_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
loss/huber_loss/Const�
loss/huber_loss/LessEqual	LessEqualloss/huber_loss/Abs:y:0loss/huber_loss/Cast/x:output:0*
T0*
_output_shapes	
:�2
loss/huber_loss/LessEquals
loss/huber_loss/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
loss/huber_loss/Pow/y�
loss/huber_loss/PowPowloss/huber_loss/Sub:z:0loss/huber_loss/Pow/y:output:0*
T0*
_output_shapes	
:�2
loss/huber_loss/Pow�
loss/huber_loss/mulMulloss/huber_loss/Const:output:0loss/huber_loss/Pow:z:0*
T0*
_output_shapes	
:�2
loss/huber_loss/mulw
loss/huber_loss/Pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
loss/huber_loss/Pow_1/y�
loss/huber_loss/Pow_1Powloss/huber_loss/Cast/x:output:0 loss/huber_loss/Pow_1/y:output:0*
T0*
_output_shapes
: 2
loss/huber_loss/Pow_1�
loss/huber_loss/mul_1Mulloss/huber_loss/Const:output:0loss/huber_loss/Pow_1:z:0*
T0*
_output_shapes
: 2
loss/huber_loss/mul_1�
loss/huber_loss/sub_1Subloss/huber_loss/Abs:y:0loss/huber_loss/Cast/x:output:0*
T0*
_output_shapes	
:�2
loss/huber_loss/sub_1�
loss/huber_loss/mul_2Mulloss/huber_loss/Cast/x:output:0loss/huber_loss/sub_1:z:0*
T0*
_output_shapes	
:�2
loss/huber_loss/mul_2�
loss/huber_loss/addAddV2loss/huber_loss/mul_1:z:0loss/huber_loss/mul_2:z:0*
T0*
_output_shapes	
:�2
loss/huber_loss/add�
loss/huber_loss/SelectV2SelectV2loss/huber_loss/LessEqual:z:0loss/huber_loss/mul:z:0loss/huber_loss/add:z:0*
T0*
_output_shapes	
:�2
loss/huber_loss/SelectV2�
&loss/huber_loss/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&loss/huber_loss/Mean/reduction_indices�
loss/huber_loss/MeanMean!loss/huber_loss/SelectV2:output:0/loss/huber_loss/Mean/reduction_indices:output:0*
T0*
_output_shapes
: 2
loss/huber_loss/Mean�
#loss/huber_loss/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2%
#loss/huber_loss/weighted_loss/Const�
!loss/huber_loss/weighted_loss/MulMulloss/huber_loss/Mean:output:0,loss/huber_loss/weighted_loss/Const:output:0*
T0*
_output_shapes
: 2#
!loss/huber_loss/weighted_loss/Mul{

loss/mul_4Mulloss/Cast:y:0%loss/huber_loss/weighted_loss/Mul:z:0*
T0*
_output_shapes	
:�2

loss/mul_4g
loss/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�2
loss/Shape_2�
loss/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
loss/strided_slice_2/stack�
loss/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
loss/strided_slice_2/stack_1�
loss/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
loss/strided_slice_2/stack_2�
loss/strided_slice_2StridedSliceloss/Shape_2:output:0#loss/strided_slice_2/stack:output:0%loss/strided_slice_2/stack_1:output:0%loss/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
loss/strided_slice_2^
loss/mul_5/yConst*
_output_shapes
: *
dtype0*
value	B :2
loss/mul_5/yv

loss/mul_5Mulloss/strided_slice_2:output:0loss/mul_5/y:output:0*
T0*
_output_shapes
: 2

loss/mul_5b
loss/Cast_1Castloss/mul_5:z:0*

DstT0*

SrcT0*
_output_shapes
: 2
loss/Cast_1b

loss/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2

loss/Consta
loss/SumSumloss/mul_4:z:0loss/Const:output:0*
T0*
_output_shapes
: 2

loss/Suml
loss/truedivRealDivloss/Sum:output:0loss/Cast_1:y:0*
T0*
_output_shapes
: 2
loss/truedive
loss/Rank/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
loss/Rank/ConstX
	loss/RankConst*
_output_shapes
: *
dtype0*
value	B :2
	loss/Rankj
loss/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
loss/range_2/startj
loss/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
loss/range_2/delta�
loss/range_2Rangeloss/range_2/start:output:0loss/Rank:output:0loss/range_2/delta:output:0*
_output_shapes
:2
loss/range_2g
loss/Sum_1/inputConst*
_output_shapes
: *
dtype0*
valueB 2
loss/Sum_1/inputr

loss/Sum_1Sumloss/Sum_1/input:output:0loss/range_2:output:0*
T0*
_output_shapes
: 2

loss/Sum_1i
loss/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
loss/truediv_1/y|
loss/truediv_1RealDivloss/Sum_1:output:0loss/truediv_1/y:output:0*
T0*
_output_shapes
: 2
loss/truediv_1h

loss/add_1AddV2loss/truediv:z:0loss/truediv_1:z:0*
T0*
_output_shapes
: 2

loss/add_1�
"Losses/td_loss/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"Losses/td_loss/write_summary/Const�
#Losses/reg_loss/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#Losses/reg_loss/write_summary/Const�
%Losses/total_loss/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2'
%Losses/total_loss/write_summary/Const�
%Variables/StopGradient/ReadVariableOpReadVariableOpBloss_qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02'
%Variables/StopGradient/ReadVariableOp�
Variables/StopGradientStopGradient-Variables/StopGradient/ReadVariableOp:value:0*
T0*
_output_shapes

:	2
Variables/StopGradient�
EVariables/QNetwork/EncodingNetwork/dense/kernel_0/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2G
EVariables/QNetwork/EncodingNetwork/dense/kernel_0/write_summary/Const�
'Variables/StopGradient_1/ReadVariableOpReadVariableOpCloss_qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Variables/StopGradient_1/ReadVariableOp�
Variables/StopGradient_1StopGradient/Variables/StopGradient_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
Variables/StopGradient_1�
CVariables/QNetwork/EncodingNetwork/dense/bias_0/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2E
CVariables/QNetwork/EncodingNetwork/dense/bias_0/write_summary/Const�
'Variables/StopGradient_2/ReadVariableOpReadVariableOpDloss_qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Variables/StopGradient_2/ReadVariableOp�
Variables/StopGradient_2StopGradient/Variables/StopGradient_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Variables/StopGradient_2�
GVariables/QNetwork/EncodingNetwork/dense_1/kernel_0/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2I
GVariables/QNetwork/EncodingNetwork/dense_1/kernel_0/write_summary/Const�
'Variables/StopGradient_3/ReadVariableOpReadVariableOpEloss_qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Variables/StopGradient_3/ReadVariableOp�
Variables/StopGradient_3StopGradient/Variables/StopGradient_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
Variables/StopGradient_3�
EVariables/QNetwork/EncodingNetwork/dense_1/bias_0/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2G
EVariables/QNetwork/EncodingNetwork/dense_1/bias_0/write_summary/Const�
'Variables/StopGradient_4/ReadVariableOpReadVariableOp4loss_qnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Variables/StopGradient_4/ReadVariableOp�
Variables/StopGradient_4StopGradient/Variables/StopGradient_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
Variables/StopGradient_4�
7Variables/QNetwork/dense_2/kernel_0/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 29
7Variables/QNetwork/dense_2/kernel_0/write_summary/Const�
'Variables/StopGradient_5/ReadVariableOpReadVariableOp5loss_qnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'Variables/StopGradient_5/ReadVariableOp�
Variables/StopGradient_5StopGradient/Variables/StopGradient_5/ReadVariableOp:value:0*
T0*
_output_shapes
:2
Variables/StopGradient_5�
5Variables/QNetwork/dense_2/bias_0/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 27
5Variables/QNetwork/dense_2/bias_0/write_summary/Constw

loss/sub_1Subloss/GatherNd:output:0loss/GatherNd_1:output:0*
T0*
_output_shapes	
:�2

loss/sub_1~
loss/td_error/StopGradientStopGradientloss/mul_3:z:0*
T0*
_output_shapes	
:�2
loss/td_error/StopGradient�
+loss/td_error/histogram/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2-
+loss/td_error/histogram/write_summary/Constt
loss/td_error/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
loss/td_error/Const
loss/td_error/MeanMeanloss/mul_3:z:0loss/td_error/Const:output:0*
T0*
_output_shapes
: 2
loss/td_error/Mean�
(loss/td_error/mean_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2*
(loss/td_error/mean_1/write_summary/Constc
loss/td_error/AbsAbsloss/mul_3:z:0*
T0*
_output_shapes	
:�2
loss/td_error/Absx
loss/td_error/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/td_error/Const_1�
loss/td_error/Mean_2Meanloss/td_error/Abs:y:0loss/td_error/Const_1:output:0*
T0*
_output_shapes
: 2
loss/td_error/Mean_2�
*loss/td_error/mean_abs/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2,
*loss/td_error/mean_abs/write_summary/Constx
loss/td_error/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/td_error/Const_2~
loss/td_error/MaxMaxloss/mul_3:z:0loss/td_error/Const_2:output:0*
T0*
_output_shapes
: 2
loss/td_error/Max�
'loss/td_error/max_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2)
'loss/td_error/max_1/write_summary/Constx
loss/td_error/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/td_error/Const_3~
loss/td_error/MinMinloss/mul_3:z:0loss/td_error/Const_3:output:0*
T0*
_output_shapes
: 2
loss/td_error/Min�
'loss/td_error/min_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2)
'loss/td_error/min_1/write_summary/Const|
loss/td_loss/StopGradientStopGradientloss/mul_4:z:0*
T0*
_output_shapes	
:�2
loss/td_loss/StopGradient�
*loss/td_loss/histogram/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2,
*loss/td_loss/histogram/write_summary/Constr
loss/td_loss/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
loss/td_loss/Const|
loss/td_loss/MeanMeanloss/mul_4:z:0loss/td_loss/Const:output:0*
T0*
_output_shapes
: 2
loss/td_loss/Mean�
'loss/td_loss/mean_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2)
'loss/td_loss/mean_1/write_summary/Consta
loss/td_loss/AbsAbsloss/mul_4:z:0*
T0*
_output_shapes	
:�2
loss/td_loss/Absv
loss/td_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/td_loss/Const_1�
loss/td_loss/Mean_2Meanloss/td_loss/Abs:y:0loss/td_loss/Const_1:output:0*
T0*
_output_shapes
: 2
loss/td_loss/Mean_2�
)loss/td_loss/mean_abs/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)loss/td_loss/mean_abs/write_summary/Constv
loss/td_loss/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/td_loss/Const_2{
loss/td_loss/MaxMaxloss/mul_4:z:0loss/td_loss/Const_2:output:0*
T0*
_output_shapes
: 2
loss/td_loss/Max�
&loss/td_loss/max_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2(
&loss/td_loss/max_1/write_summary/Constv
loss/td_loss/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/td_loss/Const_3{
loss/td_loss/MinMinloss/mul_4:z:0loss/td_loss/Const_3:output:0*
T0*
_output_shapes
: 2
loss/td_loss/Min�
&loss/td_loss/min_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2(
&loss/td_loss/min_1/write_summary/Const�
loss/q_values/StopGradientStopGradientloss/GatherNd:output:0*
T0*
_output_shapes	
:�2
loss/q_values/StopGradient�
+loss/q_values/histogram/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2-
+loss/q_values/histogram/write_summary/Constt
loss/q_values/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
loss/q_values/Const�
loss/q_values/MeanMeanloss/GatherNd:output:0loss/q_values/Const:output:0*
T0*
_output_shapes
: 2
loss/q_values/Mean�
(loss/q_values/mean_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2*
(loss/q_values/mean_1/write_summary/Constk
loss/q_values/AbsAbsloss/GatherNd:output:0*
T0*
_output_shapes	
:�2
loss/q_values/Absx
loss/q_values/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/q_values/Const_1�
loss/q_values/Mean_2Meanloss/q_values/Abs:y:0loss/q_values/Const_1:output:0*
T0*
_output_shapes
: 2
loss/q_values/Mean_2�
*loss/q_values/mean_abs/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2,
*loss/q_values/mean_abs/write_summary/Constx
loss/q_values/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/q_values/Const_2�
loss/q_values/MaxMaxloss/GatherNd:output:0loss/q_values/Const_2:output:0*
T0*
_output_shapes
: 2
loss/q_values/Max�
'loss/q_values/max_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2)
'loss/q_values/max_1/write_summary/Constx
loss/q_values/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/q_values/Const_3�
loss/q_values/MinMinloss/GatherNd:output:0loss/q_values/Const_3:output:0*
T0*
_output_shapes
: 2
loss/q_values/Min�
'loss/q_values/min_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2)
'loss/q_values/min_1/write_summary/Const�
loss/next_q_values/StopGradientStopGradientloss/GatherNd_1:output:0*
T0*
_output_shapes	
:�2!
loss/next_q_values/StopGradient�
0loss/next_q_values/histogram/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 22
0loss/next_q_values/histogram/write_summary/Const~
loss/next_q_values/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
loss/next_q_values/Const�
loss/next_q_values/MeanMeanloss/GatherNd_1:output:0!loss/next_q_values/Const:output:0*
T0*
_output_shapes
: 2
loss/next_q_values/Mean�
-loss/next_q_values/mean_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-loss/next_q_values/mean_1/write_summary/Constw
loss/next_q_values/AbsAbsloss/GatherNd_1:output:0*
T0*
_output_shapes	
:�2
loss/next_q_values/Abs�
loss/next_q_values/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/next_q_values/Const_1�
loss/next_q_values/Mean_2Meanloss/next_q_values/Abs:y:0#loss/next_q_values/Const_1:output:0*
T0*
_output_shapes
: 2
loss/next_q_values/Mean_2�
/loss/next_q_values/mean_abs/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 21
/loss/next_q_values/mean_abs/write_summary/Const�
loss/next_q_values/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/next_q_values/Const_2�
loss/next_q_values/MaxMaxloss/GatherNd_1:output:0#loss/next_q_values/Const_2:output:0*
T0*
_output_shapes
: 2
loss/next_q_values/Max�
,loss/next_q_values/max_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2.
,loss/next_q_values/max_1/write_summary/Const�
loss/next_q_values/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/next_q_values/Const_3�
loss/next_q_values/MinMinloss/GatherNd_1:output:0#loss/next_q_values/Const_3:output:0*
T0*
_output_shapes
: 2
loss/next_q_values/Min�
,loss/next_q_values/min_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2.
,loss/next_q_values/min_1/write_summary/Const�
loss/diff_q_values/StopGradientStopGradientloss/sub_1:z:0*
T0*
_output_shapes	
:�2!
loss/diff_q_values/StopGradient�
0loss/diff_q_values/histogram/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 22
0loss/diff_q_values/histogram/write_summary/Const~
loss/diff_q_values/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
loss/diff_q_values/Const�
loss/diff_q_values/MeanMeanloss/sub_1:z:0!loss/diff_q_values/Const:output:0*
T0*
_output_shapes
: 2
loss/diff_q_values/Mean�
-loss/diff_q_values/mean_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2/
-loss/diff_q_values/mean_1/write_summary/Constm
loss/diff_q_values/AbsAbsloss/sub_1:z:0*
T0*
_output_shapes	
:�2
loss/diff_q_values/Abs�
loss/diff_q_values/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/diff_q_values/Const_1�
loss/diff_q_values/Mean_2Meanloss/diff_q_values/Abs:y:0#loss/diff_q_values/Const_1:output:0*
T0*
_output_shapes
: 2
loss/diff_q_values/Mean_2�
/loss/diff_q_values/mean_abs/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 21
/loss/diff_q_values/mean_abs/write_summary/Const�
loss/diff_q_values/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/diff_q_values/Const_2�
loss/diff_q_values/MaxMaxloss/sub_1:z:0#loss/diff_q_values/Const_2:output:0*
T0*
_output_shapes
: 2
loss/diff_q_values/Max�
,loss/diff_q_values/max_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2.
,loss/diff_q_values/max_1/write_summary/Const�
loss/diff_q_values/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2
loss/diff_q_values/Const_3�
loss/diff_q_values/MinMinloss/sub_1:z:0#loss/diff_q_values/Const_3:output:0*
T0*
_output_shapes
: 2
loss/diff_q_values/Min�
,loss/diff_q_values/min_1/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2.
,loss/diff_q_values/min_1/write_summary/Const�
CheckNumericsCheckNumericsloss/add_1:z:0*
T0*
_output_shapes
: *
messageLoss is inf or nan2
CheckNumericsQ
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
ones�
 gradient_tape/loss/truediv/ShapeConst*
_output_shapes
: *
dtype0*
valueB 2"
 gradient_tape/loss/truediv/Shape�
"gradient_tape/loss/truediv/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2$
"gradient_tape/loss/truediv/Shape_1�
0gradient_tape/loss/truediv/BroadcastGradientArgsBroadcastGradientArgs)gradient_tape/loss/truediv/Shape:output:0+gradient_tape/loss/truediv/Shape_1:output:0*2
_output_shapes 
:���������:���������22
0gradient_tape/loss/truediv/BroadcastGradientArgs�
"gradient_tape/loss/truediv/RealDivRealDivones:output:0loss/Cast_1:y:0*
T0*
_output_shapes
: 2$
"gradient_tape/loss/truediv/RealDiv�
gradient_tape/loss/truediv/SumSum&gradient_tape/loss/truediv/RealDiv:z:05gradient_tape/loss/truediv/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
: 2 
gradient_tape/loss/truediv/Sum�
"gradient_tape/loss/truediv/ReshapeReshape'gradient_tape/loss/truediv/Sum:output:0)gradient_tape/loss/truediv/Shape:output:0*
T0*
_output_shapes
: 2$
"gradient_tape/loss/truediv/Reshape{
gradient_tape/loss/truediv/NegNegloss/Sum:output:0*
T0*
_output_shapes
: 2 
gradient_tape/loss/truediv/Neg�
$gradient_tape/loss/truediv/RealDiv_1RealDiv"gradient_tape/loss/truediv/Neg:y:0loss/Cast_1:y:0*
T0*
_output_shapes
: 2&
$gradient_tape/loss/truediv/RealDiv_1�
$gradient_tape/loss/truediv/RealDiv_2RealDiv(gradient_tape/loss/truediv/RealDiv_1:z:0loss/Cast_1:y:0*
T0*
_output_shapes
: 2&
$gradient_tape/loss/truediv/RealDiv_2�
gradient_tape/loss/truediv/mulMulones:output:0(gradient_tape/loss/truediv/RealDiv_2:z:0*
T0*
_output_shapes
: 2 
gradient_tape/loss/truediv/mul�
 gradient_tape/loss/truediv/Sum_1Sum"gradient_tape/loss/truediv/mul:z:05gradient_tape/loss/truediv/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
: 2"
 gradient_tape/loss/truediv/Sum_1�
$gradient_tape/loss/truediv/Reshape_1Reshape)gradient_tape/loss/truediv/Sum_1:output:0+gradient_tape/loss/truediv/Shape_1:output:0*
T0*
_output_shapes
: 2&
$gradient_tape/loss/truediv/Reshape_1�
 gradient_tape/loss/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:2"
 gradient_tape/loss/Reshape/shape�
gradient_tape/loss/ReshapeReshape+gradient_tape/loss/truediv/Reshape:output:0)gradient_tape/loss/Reshape/shape:output:0*
T0*
_output_shapes
:2
gradient_tape/loss/Reshape
gradient_tape/loss/ConstConst*
_output_shapes
:*
dtype0*
valueB:�2
gradient_tape/loss/Const�
gradient_tape/loss/TileTile#gradient_tape/loss/Reshape:output:0!gradient_tape/loss/Const:output:0*
T0*
_output_shapes	
:�2
gradient_tape/loss/Tile�
1gradient_tape/loss/mul_4/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB:�23
1gradient_tape/loss/mul_4/BroadcastGradientArgs/s0�
1gradient_tape/loss/mul_4/BroadcastGradientArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 23
1gradient_tape/loss/mul_4/BroadcastGradientArgs/s1�
.gradient_tape/loss/mul_4/BroadcastGradientArgsBroadcastGradientArgs:gradient_tape/loss/mul_4/BroadcastGradientArgs/s0:output:0:gradient_tape/loss/mul_4/BroadcastGradientArgs/s1:output:0*2
_output_shapes 
:���������:���������20
.gradient_tape/loss/mul_4/BroadcastGradientArgs�
gradient_tape/loss/mul_4/MulMulloss/Cast:y:0 gradient_tape/loss/Tile:output:0*
T0*
_output_shapes	
:�2
gradient_tape/loss/mul_4/Mul�
.gradient_tape/loss/mul_4/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 20
.gradient_tape/loss/mul_4/Sum/reduction_indices�
gradient_tape/loss/mul_4/SumSum gradient_tape/loss/mul_4/Mul:z:07gradient_tape/loss/mul_4/Sum/reduction_indices:output:0*
T0*
_output_shapes
: 2
gradient_tape/loss/mul_4/Sum�
&gradient_tape/loss/mul_4/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB 2(
&gradient_tape/loss/mul_4/Reshape/shape�
(gradient_tape/loss/mul_4/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB 2*
(gradient_tape/loss/mul_4/Reshape/shape_1�
 gradient_tape/loss/mul_4/ReshapeReshape%gradient_tape/loss/mul_4/Sum:output:01gradient_tape/loss/mul_4/Reshape/shape_1:output:0*
T0*
_output_shapes
: 2"
 gradient_tape/loss/mul_4/Reshape�
/gradient_tape/loss/huber_loss/weighted_loss/MulMul)gradient_tape/loss/mul_4/Reshape:output:0,loss/huber_loss/weighted_loss/Const:output:0*
T0*
_output_shapes
: 21
/gradient_tape/loss/huber_loss/weighted_loss/Mul�
'gradient_tape/loss/huber_loss/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB:2)
'gradient_tape/loss/huber_loss/Maximum/x�
'gradient_tape/loss/huber_loss/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'gradient_tape/loss/huber_loss/Maximum/y�
%gradient_tape/loss/huber_loss/MaximumMaximum0gradient_tape/loss/huber_loss/Maximum/x:output:00gradient_tape/loss/huber_loss/Maximum/y:output:0*
T0*
_output_shapes
:2'
%gradient_tape/loss/huber_loss/Maximum�
(gradient_tape/loss/huber_loss/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB:�2*
(gradient_tape/loss/huber_loss/floordiv/x�
&gradient_tape/loss/huber_loss/floordivFloorDiv1gradient_tape/loss/huber_loss/floordiv/x:output:0)gradient_tape/loss/huber_loss/Maximum:z:0*
T0*
_output_shapes
:2(
&gradient_tape/loss/huber_loss/floordiv�
+gradient_tape/loss/huber_loss/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+gradient_tape/loss/huber_loss/Reshape/shape�
%gradient_tape/loss/huber_loss/ReshapeReshape3gradient_tape/loss/huber_loss/weighted_loss/Mul:z:04gradient_tape/loss/huber_loss/Reshape/shape:output:0*
T0*
_output_shapes
:2'
%gradient_tape/loss/huber_loss/Reshape�
,gradient_tape/loss/huber_loss/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB:�2.
,gradient_tape/loss/huber_loss/Tile/multiples�
"gradient_tape/loss/huber_loss/TileTile.gradient_tape/loss/huber_loss/Reshape:output:05gradient_tape/loss/huber_loss/Tile/multiples:output:0*
T0*
_output_shapes	
:�2$
"gradient_tape/loss/huber_loss/Tile�
#gradient_tape/loss/huber_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   C2%
#gradient_tape/loss/huber_loss/Const�
%gradient_tape/loss/huber_loss/truedivRealDiv+gradient_tape/loss/huber_loss/Tile:output:0,gradient_tape/loss/huber_loss/Const:output:0*
T0*
_output_shapes	
:�2'
%gradient_tape/loss/huber_loss/truediv�
#gradient_tape/loss/huber_loss/zerosConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#gradient_tape/loss/huber_loss/zeros�
&gradient_tape/loss/huber_loss/SelectV2SelectV2loss/huber_loss/LessEqual:z:0)gradient_tape/loss/huber_loss/truediv:z:0,gradient_tape/loss/huber_loss/zeros:output:0*
T0*
_output_shapes	
:�2(
&gradient_tape/loss/huber_loss/SelectV2�
#gradient_tape/loss/huber_loss/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�2%
#gradient_tape/loss/huber_loss/Shape�
%gradient_tape/loss/huber_loss/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�2'
%gradient_tape/loss/huber_loss/Shape_1�
3gradient_tape/loss/huber_loss/BroadcastGradientArgsBroadcastGradientArgs,gradient_tape/loss/huber_loss/Shape:output:0.gradient_tape/loss/huber_loss/Shape_1:output:0*2
_output_shapes 
:���������:���������25
3gradient_tape/loss/huber_loss/BroadcastGradientArgs�
!gradient_tape/loss/huber_loss/SumSum/gradient_tape/loss/huber_loss/SelectV2:output:08gradient_tape/loss/huber_loss/BroadcastGradientArgs:r0:0*
T0*
_output_shapes	
:�*
	keep_dims(2#
!gradient_tape/loss/huber_loss/Sum�
'gradient_tape/loss/huber_loss/Reshape_1Reshape*gradient_tape/loss/huber_loss/Sum:output:0,gradient_tape/loss/huber_loss/Shape:output:0*
T0*
_output_shapes	
:�2)
'gradient_tape/loss/huber_loss/Reshape_1�
(gradient_tape/loss/huber_loss/SelectV2_1SelectV2loss/huber_loss/LessEqual:z:0,gradient_tape/loss/huber_loss/zeros:output:0)gradient_tape/loss/huber_loss/truediv:z:0*
T0*
_output_shapes	
:�2*
(gradient_tape/loss/huber_loss/SelectV2_1�
%gradient_tape/loss/huber_loss/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�2'
%gradient_tape/loss/huber_loss/Shape_2�
5gradient_tape/loss/huber_loss/BroadcastGradientArgs_1BroadcastGradientArgs.gradient_tape/loss/huber_loss/Shape_2:output:0.gradient_tape/loss/huber_loss/Shape_1:output:0*2
_output_shapes 
:���������:���������27
5gradient_tape/loss/huber_loss/BroadcastGradientArgs_1�
#gradient_tape/loss/huber_loss/Sum_1Sum1gradient_tape/loss/huber_loss/SelectV2_1:output:0:gradient_tape/loss/huber_loss/BroadcastGradientArgs_1:r0:0*
T0*
_output_shapes	
:�*
	keep_dims(2%
#gradient_tape/loss/huber_loss/Sum_1�
'gradient_tape/loss/huber_loss/Reshape_2Reshape,gradient_tape/loss/huber_loss/Sum_1:output:0.gradient_tape/loss/huber_loss/Shape_2:output:0*
T0*
_output_shapes	
:�2)
'gradient_tape/loss/huber_loss/Reshape_2�
:gradient_tape/loss/huber_loss/mul/BroadcastGradientArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 2<
:gradient_tape/loss/huber_loss/mul/BroadcastGradientArgs/s0�
<gradient_tape/loss/huber_loss/mul/BroadcastGradientArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB 2>
<gradient_tape/loss/huber_loss/mul/BroadcastGradientArgs/s0_1�
:gradient_tape/loss/huber_loss/mul/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*
valueB:�2<
:gradient_tape/loss/huber_loss/mul/BroadcastGradientArgs/s1�
7gradient_tape/loss/huber_loss/mul/BroadcastGradientArgsBroadcastGradientArgsEgradient_tape/loss/huber_loss/mul/BroadcastGradientArgs/s0_1:output:0Cgradient_tape/loss/huber_loss/mul/BroadcastGradientArgs/s1:output:0*2
_output_shapes 
:���������:���������29
7gradient_tape/loss/huber_loss/mul/BroadcastGradientArgs�
%gradient_tape/loss/huber_loss/mul/MulMulloss/huber_loss/Const:output:00gradient_tape/loss/huber_loss/Reshape_1:output:0*
T0*
_output_shapes	
:�2'
%gradient_tape/loss/huber_loss/mul/Mul�
%gradient_tape/loss/huber_loss/Pow/mulMul)gradient_tape/loss/huber_loss/mul/Mul:z:0loss/huber_loss/Pow/y:output:0*
T0*
_output_shapes	
:�2'
%gradient_tape/loss/huber_loss/Pow/mul�
'gradient_tape/loss/huber_loss/Pow/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2)
'gradient_tape/loss/huber_loss/Pow/sub/y�
%gradient_tape/loss/huber_loss/Pow/subSubloss/huber_loss/Pow/y:output:00gradient_tape/loss/huber_loss/Pow/sub/y:output:0*
T0*
_output_shapes
: 2'
%gradient_tape/loss/huber_loss/Pow/sub�
%gradient_tape/loss/huber_loss/Pow/PowPowloss/huber_loss/Sub:z:0)gradient_tape/loss/huber_loss/Pow/sub:z:0*
T0*
_output_shapes	
:�2'
%gradient_tape/loss/huber_loss/Pow/Pow�
'gradient_tape/loss/huber_loss/Pow/mul_1Mul)gradient_tape/loss/huber_loss/Pow/mul:z:0)gradient_tape/loss/huber_loss/Pow/Pow:z:0*
T0*
_output_shapes	
:�2)
'gradient_tape/loss/huber_loss/Pow/mul_1�
'gradient_tape/loss/huber_loss/mul_2/MulMulloss/huber_loss/Cast/x:output:00gradient_tape/loss/huber_loss/Reshape_2:output:0*
T0*
_output_shapes	
:�2)
'gradient_tape/loss/huber_loss/mul_2/Mul�
&gradient_tape/loss/huber_loss/Abs/SignSignloss/huber_loss/Sub:z:0*
T0*
_output_shapes	
:�2(
&gradient_tape/loss/huber_loss/Abs/Sign�
%gradient_tape/loss/huber_loss/Abs/mulMul+gradient_tape/loss/huber_loss/mul_2/Mul:z:0*gradient_tape/loss/huber_loss/Abs/Sign:y:0*
T0*
_output_shapes	
:�2'
%gradient_tape/loss/huber_loss/Abs/mul�
AddNAddN+gradient_tape/loss/huber_loss/Pow/mul_1:z:0)gradient_tape/loss/huber_loss/Abs/mul:z:0*
N*
T0*
_output_shapes	
:�2
AddN
!gradient_tape/loss/huber_loss/NegNeg
AddN:sum:0*
T0*
_output_shapes	
:�2#
!gradient_tape/loss/huber_loss/Neg�
gradient_tape/loss/ShapeConst*
_output_shapes
:*
dtype0*
valueB"�      2
gradient_tape/loss/Shape�
gradient_tape/loss/ScatterNd	ScatterNdloss/concat:output:0
AddN:sum:0!gradient_tape/loss/Shape:output:0*
T0*
Tindices0*
_output_shapes
:	�2
gradient_tape/loss/ScatterNd�
7gradient_tape/loss/QNetwork/dense_2/BiasAdd/BiasAddGradBiasAddGrad%gradient_tape/loss/ScatterNd:output:0*
T0*
_output_shapes
:29
7gradient_tape/loss/QNetwork/dense_2/BiasAdd/BiasAddGrad�
*gradient_tape/loss/QNetwork/dense_2/MatMulMatMul%gradient_tape/loss/ScatterNd:output:03loss/QNetwork/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�*
transpose_b(2,
*gradient_tape/loss/QNetwork/dense_2/MatMul�
,gradient_tape/loss/QNetwork/dense_2/MatMul_1MatMul8loss/QNetwork/EncodingNetwork/dense_1/IdentityN:output:0%gradient_tape/loss/ScatterNd:output:0*
T0*
_output_shapes

:*
transpose_a(2.
,gradient_tape/loss/QNetwork/dense_2/MatMul_1
zeros/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"�      2
zeros/shape_as_tensor_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/shape_as_tensor:output:0zeros/Const:output:0*
T0*
_output_shapes
:	�2
zeros�
SigmoidSigmoid6loss/QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0+^gradient_tape/loss/QNetwork/dense_2/MatMul*
T0*
_output_shapes
:	�2	
SigmoidS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	�2
sub�
mul_4Mul6loss/QNetwork/EncodingNetwork/dense_1/BiasAdd:output:0sub:z:0*
T0*
_output_shapes
:	�2
mul_4S
add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
add/xX
addAddV2add/x:output:0	mul_4:z:0*
T0*
_output_shapes
:	�2
addU
mul_5MulSigmoid:y:0add:z:0*
T0*
_output_shapes
:	�2
mul_5�
mul_6Mul4gradient_tape/loss/QNetwork/dense_2/MatMul:product:0	mul_5:z:0*
T0*
_output_shapes
:	�2
mul_6�
Ggradient_tape/loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/BiasAddGradBiasAddGrad	mul_6:z:0*
T0*
_output_shapes
:2I
Ggradient_tape/loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/BiasAddGrad�
:gradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMulMatMul	mul_6:z:0Closs/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�*
transpose_b(2<
:gradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul�
<gradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul_1MatMul6loss/QNetwork/EncodingNetwork/dense/IdentityN:output:0	mul_6:z:0*
T0*
_output_shapes

:*
transpose_a(2>
<gradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul_1�
zeros_1/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"�      2
zeros_1/shape_as_tensorc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fill zeros_1/shape_as_tensor:output:0zeros_1/Const:output:0*
T0*
_output_shapes
:	�2	
zeros_1�
	Sigmoid_1Sigmoid4loss/QNetwork/EncodingNetwork/dense/BiasAdd:output:0;^gradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul*
T0*
_output_shapes
:	�2
	Sigmoid_1W
sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
sub_1/x`
sub_1Subsub_1/x:output:0Sigmoid_1:y:0*
T0*
_output_shapes
:	�2
sub_1�
mul_7Mul4loss/QNetwork/EncodingNetwork/dense/BiasAdd:output:0	sub_1:z:0*
T0*
_output_shapes
:	�2
mul_7W
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2	
add_1/x^
add_1AddV2add_1/x:output:0	mul_7:z:0*
T0*
_output_shapes
:	�2
add_1Y
mul_8MulSigmoid_1:y:0	add_1:z:0*
T0*
_output_shapes
:	�2
mul_8�
mul_9MulDgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul:product:0	mul_8:z:0*
T0*
_output_shapes
:	�2
mul_9�
Egradient_tape/loss/QNetwork/EncodingNetwork/dense/BiasAdd/BiasAddGradBiasAddGrad	mul_9:z:0*
T0*
_output_shapes
:2G
Egradient_tape/loss/QNetwork/EncodingNetwork/dense/BiasAdd/BiasAddGrad�
8gradient_tape/loss/QNetwork/EncodingNetwork/dense/MatMulMatMul6loss/QNetwork/EncodingNetwork/flatten/Reshape:output:0	mul_9:z:0*
T0*
_output_shapes

:	*
transpose_a(2:
8gradient_tape/loss/QNetwork/EncodingNetwork/dense/MatMul�
*summarize_vars/StopGradient/ReadVariableOpReadVariableOpBloss_qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02,
*summarize_vars/StopGradient/ReadVariableOp�
summarize_vars/StopGradientStopGradient2summarize_vars/StopGradient/ReadVariableOp:value:0*
T0*
_output_shapes

:	2
summarize_vars/StopGradient�
Psummarize_vars/QNetwork/EncodingNetwork/dense/kernel_0_value/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2R
Psummarize_vars/QNetwork/EncodingNetwork/dense/kernel_0_value/write_summary/Const�
)summarize_vars/global_norm/ReadVariableOpReadVariableOpBloss_qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02+
)summarize_vars/global_norm/ReadVariableOp�
!summarize_vars/global_norm/L2LossL2Loss1summarize_vars/global_norm/ReadVariableOp:value:0*
T0*<
_class2
0.loc:@summarize_vars/global_norm/ReadVariableOp*
_output_shapes
: 2#
!summarize_vars/global_norm/L2Loss�
 summarize_vars/global_norm/stackPack*summarize_vars/global_norm/L2Loss:output:0*
N*
T0*
_output_shapes
:2"
 summarize_vars/global_norm/stack�
 summarize_vars/global_norm/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 summarize_vars/global_norm/Const�
summarize_vars/global_norm/SumSum)summarize_vars/global_norm/stack:output:0)summarize_vars/global_norm/Const:output:0*
T0*
_output_shapes
: 2 
summarize_vars/global_norm/Sum�
"summarize_vars/global_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2$
"summarize_vars/global_norm/Const_1�
summarize_vars/global_norm/mulMul'summarize_vars/global_norm/Sum:output:0+summarize_vars/global_norm/Const_1:output:0*
T0*
_output_shapes
: 2 
summarize_vars/global_norm/mul�
&summarize_vars/global_norm/global_normSqrt"summarize_vars/global_norm/mul:z:0*
T0*
_output_shapes
: 2(
&summarize_vars/global_norm/global_norm�
Usummarize_vars/QNetwork/EncodingNetwork/dense/kernel_0_value_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2W
Usummarize_vars/QNetwork/EncodingNetwork/dense/kernel_0_value_norm/write_summary/Const�
,summarize_vars/StopGradient_1/ReadVariableOpReadVariableOpCloss_qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,summarize_vars/StopGradient_1/ReadVariableOp�
summarize_vars/StopGradient_1StopGradient4summarize_vars/StopGradient_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
summarize_vars/StopGradient_1�
Nsummarize_vars/QNetwork/EncodingNetwork/dense/bias_0_value/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2P
Nsummarize_vars/QNetwork/EncodingNetwork/dense/bias_0_value/write_summary/Const�
+summarize_vars/global_norm_1/ReadVariableOpReadVariableOpCloss_qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+summarize_vars/global_norm_1/ReadVariableOp�
#summarize_vars/global_norm_1/L2LossL2Loss3summarize_vars/global_norm_1/ReadVariableOp:value:0*
T0*>
_class4
20loc:@summarize_vars/global_norm_1/ReadVariableOp*
_output_shapes
: 2%
#summarize_vars/global_norm_1/L2Loss�
"summarize_vars/global_norm_1/stackPack,summarize_vars/global_norm_1/L2Loss:output:0*
N*
T0*
_output_shapes
:2$
"summarize_vars/global_norm_1/stack�
"summarize_vars/global_norm_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"summarize_vars/global_norm_1/Const�
 summarize_vars/global_norm_1/SumSum+summarize_vars/global_norm_1/stack:output:0+summarize_vars/global_norm_1/Const:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_1/Sum�
$summarize_vars/global_norm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$summarize_vars/global_norm_1/Const_1�
 summarize_vars/global_norm_1/mulMul)summarize_vars/global_norm_1/Sum:output:0-summarize_vars/global_norm_1/Const_1:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_1/mul�
(summarize_vars/global_norm_1/global_normSqrt$summarize_vars/global_norm_1/mul:z:0*
T0*
_output_shapes
: 2*
(summarize_vars/global_norm_1/global_norm�
Ssummarize_vars/QNetwork/EncodingNetwork/dense/bias_0_value_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2U
Ssummarize_vars/QNetwork/EncodingNetwork/dense/bias_0_value_norm/write_summary/Const�
,summarize_vars/StopGradient_2/ReadVariableOpReadVariableOpDloss_qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,summarize_vars/StopGradient_2/ReadVariableOp�
summarize_vars/StopGradient_2StopGradient4summarize_vars/StopGradient_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2
summarize_vars/StopGradient_2�
Rsummarize_vars/QNetwork/EncodingNetwork/dense_1/kernel_0_value/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2T
Rsummarize_vars/QNetwork/EncodingNetwork/dense_1/kernel_0_value/write_summary/Const�
+summarize_vars/global_norm_2/ReadVariableOpReadVariableOpDloss_qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+summarize_vars/global_norm_2/ReadVariableOp�
#summarize_vars/global_norm_2/L2LossL2Loss3summarize_vars/global_norm_2/ReadVariableOp:value:0*
T0*>
_class4
20loc:@summarize_vars/global_norm_2/ReadVariableOp*
_output_shapes
: 2%
#summarize_vars/global_norm_2/L2Loss�
"summarize_vars/global_norm_2/stackPack,summarize_vars/global_norm_2/L2Loss:output:0*
N*
T0*
_output_shapes
:2$
"summarize_vars/global_norm_2/stack�
"summarize_vars/global_norm_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"summarize_vars/global_norm_2/Const�
 summarize_vars/global_norm_2/SumSum+summarize_vars/global_norm_2/stack:output:0+summarize_vars/global_norm_2/Const:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_2/Sum�
$summarize_vars/global_norm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$summarize_vars/global_norm_2/Const_1�
 summarize_vars/global_norm_2/mulMul)summarize_vars/global_norm_2/Sum:output:0-summarize_vars/global_norm_2/Const_1:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_2/mul�
(summarize_vars/global_norm_2/global_normSqrt$summarize_vars/global_norm_2/mul:z:0*
T0*
_output_shapes
: 2*
(summarize_vars/global_norm_2/global_norm�
Wsummarize_vars/QNetwork/EncodingNetwork/dense_1/kernel_0_value_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2Y
Wsummarize_vars/QNetwork/EncodingNetwork/dense_1/kernel_0_value_norm/write_summary/Const�
,summarize_vars/StopGradient_3/ReadVariableOpReadVariableOpEloss_qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,summarize_vars/StopGradient_3/ReadVariableOp�
summarize_vars/StopGradient_3StopGradient4summarize_vars/StopGradient_3/ReadVariableOp:value:0*
T0*
_output_shapes
:2
summarize_vars/StopGradient_3�
Psummarize_vars/QNetwork/EncodingNetwork/dense_1/bias_0_value/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2R
Psummarize_vars/QNetwork/EncodingNetwork/dense_1/bias_0_value/write_summary/Const�
+summarize_vars/global_norm_3/ReadVariableOpReadVariableOpEloss_qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+summarize_vars/global_norm_3/ReadVariableOp�
#summarize_vars/global_norm_3/L2LossL2Loss3summarize_vars/global_norm_3/ReadVariableOp:value:0*
T0*>
_class4
20loc:@summarize_vars/global_norm_3/ReadVariableOp*
_output_shapes
: 2%
#summarize_vars/global_norm_3/L2Loss�
"summarize_vars/global_norm_3/stackPack,summarize_vars/global_norm_3/L2Loss:output:0*
N*
T0*
_output_shapes
:2$
"summarize_vars/global_norm_3/stack�
"summarize_vars/global_norm_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"summarize_vars/global_norm_3/Const�
 summarize_vars/global_norm_3/SumSum+summarize_vars/global_norm_3/stack:output:0+summarize_vars/global_norm_3/Const:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_3/Sum�
$summarize_vars/global_norm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$summarize_vars/global_norm_3/Const_1�
 summarize_vars/global_norm_3/mulMul)summarize_vars/global_norm_3/Sum:output:0-summarize_vars/global_norm_3/Const_1:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_3/mul�
(summarize_vars/global_norm_3/global_normSqrt$summarize_vars/global_norm_3/mul:z:0*
T0*
_output_shapes
: 2*
(summarize_vars/global_norm_3/global_norm�
Usummarize_vars/QNetwork/EncodingNetwork/dense_1/bias_0_value_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2W
Usummarize_vars/QNetwork/EncodingNetwork/dense_1/bias_0_value_norm/write_summary/Const�
,summarize_vars/StopGradient_4/ReadVariableOpReadVariableOp4loss_qnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02.
,summarize_vars/StopGradient_4/ReadVariableOp�
summarize_vars/StopGradient_4StopGradient4summarize_vars/StopGradient_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
summarize_vars/StopGradient_4�
Bsummarize_vars/QNetwork/dense_2/kernel_0_value/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2D
Bsummarize_vars/QNetwork/dense_2/kernel_0_value/write_summary/Const�
+summarize_vars/global_norm_4/ReadVariableOpReadVariableOp4loss_qnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+summarize_vars/global_norm_4/ReadVariableOp�
#summarize_vars/global_norm_4/L2LossL2Loss3summarize_vars/global_norm_4/ReadVariableOp:value:0*
T0*>
_class4
20loc:@summarize_vars/global_norm_4/ReadVariableOp*
_output_shapes
: 2%
#summarize_vars/global_norm_4/L2Loss�
"summarize_vars/global_norm_4/stackPack,summarize_vars/global_norm_4/L2Loss:output:0*
N*
T0*
_output_shapes
:2$
"summarize_vars/global_norm_4/stack�
"summarize_vars/global_norm_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"summarize_vars/global_norm_4/Const�
 summarize_vars/global_norm_4/SumSum+summarize_vars/global_norm_4/stack:output:0+summarize_vars/global_norm_4/Const:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_4/Sum�
$summarize_vars/global_norm_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$summarize_vars/global_norm_4/Const_1�
 summarize_vars/global_norm_4/mulMul)summarize_vars/global_norm_4/Sum:output:0-summarize_vars/global_norm_4/Const_1:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_4/mul�
(summarize_vars/global_norm_4/global_normSqrt$summarize_vars/global_norm_4/mul:z:0*
T0*
_output_shapes
: 2*
(summarize_vars/global_norm_4/global_norm�
Gsummarize_vars/QNetwork/dense_2/kernel_0_value_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2I
Gsummarize_vars/QNetwork/dense_2/kernel_0_value_norm/write_summary/Const�
,summarize_vars/StopGradient_5/ReadVariableOpReadVariableOp5loss_qnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,summarize_vars/StopGradient_5/ReadVariableOp�
summarize_vars/StopGradient_5StopGradient4summarize_vars/StopGradient_5/ReadVariableOp:value:0*
T0*
_output_shapes
:2
summarize_vars/StopGradient_5�
@summarize_vars/QNetwork/dense_2/bias_0_value/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2B
@summarize_vars/QNetwork/dense_2/bias_0_value/write_summary/Const�
+summarize_vars/global_norm_5/ReadVariableOpReadVariableOp5loss_qnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+summarize_vars/global_norm_5/ReadVariableOp�
#summarize_vars/global_norm_5/L2LossL2Loss3summarize_vars/global_norm_5/ReadVariableOp:value:0*
T0*>
_class4
20loc:@summarize_vars/global_norm_5/ReadVariableOp*
_output_shapes
: 2%
#summarize_vars/global_norm_5/L2Loss�
"summarize_vars/global_norm_5/stackPack,summarize_vars/global_norm_5/L2Loss:output:0*
N*
T0*
_output_shapes
:2$
"summarize_vars/global_norm_5/stack�
"summarize_vars/global_norm_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"summarize_vars/global_norm_5/Const�
 summarize_vars/global_norm_5/SumSum+summarize_vars/global_norm_5/stack:output:0+summarize_vars/global_norm_5/Const:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_5/Sum�
$summarize_vars/global_norm_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$summarize_vars/global_norm_5/Const_1�
 summarize_vars/global_norm_5/mulMul)summarize_vars/global_norm_5/Sum:output:0-summarize_vars/global_norm_5/Const_1:output:0*
T0*
_output_shapes
: 2"
 summarize_vars/global_norm_5/mul�
(summarize_vars/global_norm_5/global_normSqrt$summarize_vars/global_norm_5/mul:z:0*
T0*
_output_shapes
: 2*
(summarize_vars/global_norm_5/global_norm�
Esummarize_vars/QNetwork/dense_2/bias_0_value_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2G
Esummarize_vars/QNetwork/dense_2/bias_0_value_norm/write_summary/Const�
summarize_grads/StopGradientStopGradientBgradient_tape/loss/QNetwork/EncodingNetwork/dense/MatMul:product:0*
T0*
_output_shapes

:	2
summarize_grads/StopGradient�
Tsummarize_grads/QNetwork/EncodingNetwork/dense/kernel_0_gradient/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2V
Tsummarize_grads/QNetwork/EncodingNetwork/dense/kernel_0_gradient/write_summary/Const�
"summarize_grads/global_norm/L2LossL2LossBgradient_tape/loss/QNetwork/EncodingNetwork/dense/MatMul:product:0*
T0*K
_classA
?=loc:@gradient_tape/loss/QNetwork/EncodingNetwork/dense/MatMul*
_output_shapes
: 2$
"summarize_grads/global_norm/L2Loss�
!summarize_grads/global_norm/stackPack+summarize_grads/global_norm/L2Loss:output:0*
N*
T0*
_output_shapes
:2#
!summarize_grads/global_norm/stack�
!summarize_grads/global_norm/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!summarize_grads/global_norm/Const�
summarize_grads/global_norm/SumSum*summarize_grads/global_norm/stack:output:0*summarize_grads/global_norm/Const:output:0*
T0*
_output_shapes
: 2!
summarize_grads/global_norm/Sum�
#summarize_grads/global_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2%
#summarize_grads/global_norm/Const_1�
summarize_grads/global_norm/mulMul(summarize_grads/global_norm/Sum:output:0,summarize_grads/global_norm/Const_1:output:0*
T0*
_output_shapes
: 2!
summarize_grads/global_norm/mul�
'summarize_grads/global_norm/global_normSqrt#summarize_grads/global_norm/mul:z:0*
T0*
_output_shapes
: 2)
'summarize_grads/global_norm/global_norm�
Ysummarize_grads/QNetwork/EncodingNetwork/dense/kernel_0_gradient_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2[
Ysummarize_grads/QNetwork/EncodingNetwork/dense/kernel_0_gradient_norm/write_summary/Const�
summarize_grads/StopGradient_1StopGradientNgradient_tape/loss/QNetwork/EncodingNetwork/dense/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:2 
summarize_grads/StopGradient_1�
Rsummarize_grads/QNetwork/EncodingNetwork/dense/bias_0_gradient/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2T
Rsummarize_grads/QNetwork/EncodingNetwork/dense/bias_0_gradient/write_summary/Const�
$summarize_grads/global_norm_1/L2LossL2LossNgradient_tape/loss/QNetwork/EncodingNetwork/dense/BiasAdd/BiasAddGrad:output:0*
T0*X
_classN
LJloc:@gradient_tape/loss/QNetwork/EncodingNetwork/dense/BiasAdd/BiasAddGrad*
_output_shapes
: 2&
$summarize_grads/global_norm_1/L2Loss�
#summarize_grads/global_norm_1/stackPack-summarize_grads/global_norm_1/L2Loss:output:0*
N*
T0*
_output_shapes
:2%
#summarize_grads/global_norm_1/stack�
#summarize_grads/global_norm_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#summarize_grads/global_norm_1/Const�
!summarize_grads/global_norm_1/SumSum,summarize_grads/global_norm_1/stack:output:0,summarize_grads/global_norm_1/Const:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_1/Sum�
%summarize_grads/global_norm_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%summarize_grads/global_norm_1/Const_1�
!summarize_grads/global_norm_1/mulMul*summarize_grads/global_norm_1/Sum:output:0.summarize_grads/global_norm_1/Const_1:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_1/mul�
)summarize_grads/global_norm_1/global_normSqrt%summarize_grads/global_norm_1/mul:z:0*
T0*
_output_shapes
: 2+
)summarize_grads/global_norm_1/global_norm�
Wsummarize_grads/QNetwork/EncodingNetwork/dense/bias_0_gradient_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2Y
Wsummarize_grads/QNetwork/EncodingNetwork/dense/bias_0_gradient_norm/write_summary/Const�
summarize_grads/StopGradient_2StopGradientFgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul_1:product:0*
T0*
_output_shapes

:2 
summarize_grads/StopGradient_2�
Vsummarize_grads/QNetwork/EncodingNetwork/dense_1/kernel_0_gradient/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2X
Vsummarize_grads/QNetwork/EncodingNetwork/dense_1/kernel_0_gradient/write_summary/Const�
$summarize_grads/global_norm_2/L2LossL2LossFgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul_1:product:0*
T0*O
_classE
CAloc:@gradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul_1*
_output_shapes
: 2&
$summarize_grads/global_norm_2/L2Loss�
#summarize_grads/global_norm_2/stackPack-summarize_grads/global_norm_2/L2Loss:output:0*
N*
T0*
_output_shapes
:2%
#summarize_grads/global_norm_2/stack�
#summarize_grads/global_norm_2/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#summarize_grads/global_norm_2/Const�
!summarize_grads/global_norm_2/SumSum,summarize_grads/global_norm_2/stack:output:0,summarize_grads/global_norm_2/Const:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_2/Sum�
%summarize_grads/global_norm_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%summarize_grads/global_norm_2/Const_1�
!summarize_grads/global_norm_2/mulMul*summarize_grads/global_norm_2/Sum:output:0.summarize_grads/global_norm_2/Const_1:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_2/mul�
)summarize_grads/global_norm_2/global_normSqrt%summarize_grads/global_norm_2/mul:z:0*
T0*
_output_shapes
: 2+
)summarize_grads/global_norm_2/global_norm�
[summarize_grads/QNetwork/EncodingNetwork/dense_1/kernel_0_gradient_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2]
[summarize_grads/QNetwork/EncodingNetwork/dense_1/kernel_0_gradient_norm/write_summary/Const�
summarize_grads/StopGradient_3StopGradientPgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:2 
summarize_grads/StopGradient_3�
Tsummarize_grads/QNetwork/EncodingNetwork/dense_1/bias_0_gradient/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2V
Tsummarize_grads/QNetwork/EncodingNetwork/dense_1/bias_0_gradient/write_summary/Const�
$summarize_grads/global_norm_3/L2LossL2LossPgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/BiasAddGrad:output:0*
T0*Z
_classP
NLloc:@gradient_tape/loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/BiasAddGrad*
_output_shapes
: 2&
$summarize_grads/global_norm_3/L2Loss�
#summarize_grads/global_norm_3/stackPack-summarize_grads/global_norm_3/L2Loss:output:0*
N*
T0*
_output_shapes
:2%
#summarize_grads/global_norm_3/stack�
#summarize_grads/global_norm_3/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#summarize_grads/global_norm_3/Const�
!summarize_grads/global_norm_3/SumSum,summarize_grads/global_norm_3/stack:output:0,summarize_grads/global_norm_3/Const:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_3/Sum�
%summarize_grads/global_norm_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%summarize_grads/global_norm_3/Const_1�
!summarize_grads/global_norm_3/mulMul*summarize_grads/global_norm_3/Sum:output:0.summarize_grads/global_norm_3/Const_1:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_3/mul�
)summarize_grads/global_norm_3/global_normSqrt%summarize_grads/global_norm_3/mul:z:0*
T0*
_output_shapes
: 2+
)summarize_grads/global_norm_3/global_norm�
Ysummarize_grads/QNetwork/EncodingNetwork/dense_1/bias_0_gradient_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2[
Ysummarize_grads/QNetwork/EncodingNetwork/dense_1/bias_0_gradient_norm/write_summary/Const�
summarize_grads/StopGradient_4StopGradient6gradient_tape/loss/QNetwork/dense_2/MatMul_1:product:0*
T0*
_output_shapes

:2 
summarize_grads/StopGradient_4�
Fsummarize_grads/QNetwork/dense_2/kernel_0_gradient/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2H
Fsummarize_grads/QNetwork/dense_2/kernel_0_gradient/write_summary/Const�
$summarize_grads/global_norm_4/L2LossL2Loss6gradient_tape/loss/QNetwork/dense_2/MatMul_1:product:0*
T0*?
_class5
31loc:@gradient_tape/loss/QNetwork/dense_2/MatMul_1*
_output_shapes
: 2&
$summarize_grads/global_norm_4/L2Loss�
#summarize_grads/global_norm_4/stackPack-summarize_grads/global_norm_4/L2Loss:output:0*
N*
T0*
_output_shapes
:2%
#summarize_grads/global_norm_4/stack�
#summarize_grads/global_norm_4/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#summarize_grads/global_norm_4/Const�
!summarize_grads/global_norm_4/SumSum,summarize_grads/global_norm_4/stack:output:0,summarize_grads/global_norm_4/Const:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_4/Sum�
%summarize_grads/global_norm_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%summarize_grads/global_norm_4/Const_1�
!summarize_grads/global_norm_4/mulMul*summarize_grads/global_norm_4/Sum:output:0.summarize_grads/global_norm_4/Const_1:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_4/mul�
)summarize_grads/global_norm_4/global_normSqrt%summarize_grads/global_norm_4/mul:z:0*
T0*
_output_shapes
: 2+
)summarize_grads/global_norm_4/global_norm�
Ksummarize_grads/QNetwork/dense_2/kernel_0_gradient_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2M
Ksummarize_grads/QNetwork/dense_2/kernel_0_gradient_norm/write_summary/Const�
summarize_grads/StopGradient_5StopGradient@gradient_tape/loss/QNetwork/dense_2/BiasAdd/BiasAddGrad:output:0*
T0*
_output_shapes
:2 
summarize_grads/StopGradient_5�
Dsummarize_grads/QNetwork/dense_2/bias_0_gradient/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2F
Dsummarize_grads/QNetwork/dense_2/bias_0_gradient/write_summary/Const�
$summarize_grads/global_norm_5/L2LossL2Loss@gradient_tape/loss/QNetwork/dense_2/BiasAdd/BiasAddGrad:output:0*
T0*J
_class@
><loc:@gradient_tape/loss/QNetwork/dense_2/BiasAdd/BiasAddGrad*
_output_shapes
: 2&
$summarize_grads/global_norm_5/L2Loss�
#summarize_grads/global_norm_5/stackPack-summarize_grads/global_norm_5/L2Loss:output:0*
N*
T0*
_output_shapes
:2%
#summarize_grads/global_norm_5/stack�
#summarize_grads/global_norm_5/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#summarize_grads/global_norm_5/Const�
!summarize_grads/global_norm_5/SumSum,summarize_grads/global_norm_5/stack:output:0,summarize_grads/global_norm_5/Const:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_5/Sum�
%summarize_grads/global_norm_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @2'
%summarize_grads/global_norm_5/Const_1�
!summarize_grads/global_norm_5/mulMul*summarize_grads/global_norm_5/Sum:output:0.summarize_grads/global_norm_5/Const_1:output:0*
T0*
_output_shapes
: 2#
!summarize_grads/global_norm_5/mul�
)summarize_grads/global_norm_5/global_normSqrt%summarize_grads/global_norm_5/mul:z:0*
T0*
_output_shapes
: 2+
)summarize_grads/global_norm_5/global_norm�
Isummarize_grads/QNetwork/dense_2/bias_0_gradient_norm/write_summary/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2K
Isummarize_grads/QNetwork/dense_2/bias_0_gradient_norm/write_summary/Const�
RMSprop/Cast/ReadVariableOpReadVariableOp$rmsprop_cast_readvariableop_resource*
_output_shapes
: *
dtype02
RMSprop/Cast/ReadVariableOp�
RMSprop/IdentityIdentity#RMSprop/Cast/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/Identity�
RMSprop/Cast_1/ReadVariableOpReadVariableOp&rmsprop_cast_1_readvariableop_resource*
_output_shapes
: *
dtype02
RMSprop/Cast_1/ReadVariableOp�
RMSprop/Identity_1Identity%RMSprop/Cast_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/Identity_1�
RMSprop/NegNegRMSprop/Identity:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/Neg�
RMSprop/ConstConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2
RMSprop/Const�
RMSprop/Cast_2/ReadVariableOpReadVariableOp&rmsprop_cast_2_readvariableop_resource*
_output_shapes
: *
dtype02
RMSprop/Cast_2/ReadVariableOp�
RMSprop/Identity_2Identity%RMSprop/Cast_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/Identity_2�
RMSprop/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  �?2
RMSprop/sub/x�
RMSprop/subSubRMSprop/sub/x:output:0RMSprop/Identity_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: 2
RMSprop/sub�
)RMSprop/RMSprop/update/mul/ReadVariableOpReadVariableOp2rmsprop_rmsprop_update_mul_readvariableop_resource*
_output_shapes

:	*
dtype02+
)RMSprop/RMSprop/update/mul/ReadVariableOp�
RMSprop/RMSprop/update/mulMulRMSprop/Identity_1:output:01RMSprop/RMSprop/update/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/mul�
RMSprop/RMSprop/update/SquareSquareBgradient_tape/loss/QNetwork/EncodingNetwork/dense/MatMul:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/Square�
RMSprop/RMSprop/update/mul_1MulRMSprop/sub:z:0!RMSprop/RMSprop/update/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/mul_1�
RMSprop/RMSprop/update/addAddV2RMSprop/RMSprop/update/mul:z:0 RMSprop/RMSprop/update/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/add�
'RMSprop/RMSprop/update/AssignVariableOpAssignVariableOp2rmsprop_rmsprop_update_mul_readvariableop_resourceRMSprop/RMSprop/update/add:z:0*^RMSprop/RMSprop/update/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02)
'RMSprop/RMSprop/update/AssignVariableOp�
+RMSprop/RMSprop/update/mul_2/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_mul_2_readvariableop_resource*
_output_shapes

:	*
dtype02-
+RMSprop/RMSprop/update/mul_2/ReadVariableOp�
RMSprop/RMSprop/update/mul_2MulRMSprop/Identity_1:output:03RMSprop/RMSprop/update/mul_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/mul_2�
RMSprop/RMSprop/update/mul_3MulRMSprop/sub:z:0Bgradient_tape/loss/QNetwork/EncodingNetwork/dense/MatMul:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/mul_3�
RMSprop/RMSprop/update/add_1AddV2 RMSprop/RMSprop/update/mul_2:z:0 RMSprop/RMSprop/update/mul_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/add_1�
)RMSprop/RMSprop/update/AssignVariableOp_1AssignVariableOp4rmsprop_rmsprop_update_mul_2_readvariableop_resource RMSprop/RMSprop/update/add_1:z:0,^RMSprop/RMSprop/update/mul_2/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update/AssignVariableOp_1�
.RMSprop/RMSprop/update/Square_1/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_mul_2_readvariableop_resource*^RMSprop/RMSprop/update/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	*
dtype020
.RMSprop/RMSprop/update/Square_1/ReadVariableOp�
RMSprop/RMSprop/update/Square_1Square6RMSprop/RMSprop/update/Square_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2!
RMSprop/RMSprop/update/Square_1�
%RMSprop/RMSprop/update/ReadVariableOpReadVariableOp2rmsprop_rmsprop_update_mul_readvariableop_resource(^RMSprop/RMSprop/update/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	*
dtype02'
%RMSprop/RMSprop/update/ReadVariableOp�
RMSprop/RMSprop/update/subSub-RMSprop/RMSprop/update/ReadVariableOp:value:0#RMSprop/RMSprop/update/Square_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/sub�
RMSprop/RMSprop/update/mul_4MulRMSprop/Identity:output:0Bgradient_tape/loss/QNetwork/EncodingNetwork/dense/MatMul:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/mul_4�
RMSprop/RMSprop/update/SqrtSqrtRMSprop/RMSprop/update/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/Sqrt�
RMSprop/RMSprop/update/add_2AddV2RMSprop/RMSprop/update/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/add_2�
RMSprop/RMSprop/update/truedivRealDiv RMSprop/RMSprop/update/mul_4:z:0 RMSprop/RMSprop/update/add_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2 
RMSprop/RMSprop/update/truediv�
'RMSprop/RMSprop/update/ReadVariableOp_1ReadVariableOpBloss_qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02)
'RMSprop/RMSprop/update/ReadVariableOp_1�
RMSprop/RMSprop/update/sub_1Sub/RMSprop/RMSprop/update/ReadVariableOp_1:value:0"RMSprop/RMSprop/update/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes

:	2
RMSprop/RMSprop/update/sub_1�
)RMSprop/RMSprop/update/AssignVariableOp_2AssignVariableOpBloss_qnetwork_encodingnetwork_dense_matmul_readvariableop_resource RMSprop/RMSprop/update/sub_1:z:0(^RMSprop/RMSprop/update/ReadVariableOp_1&^Variables/StopGradient/ReadVariableOp:^loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp+^summarize_vars/StopGradient/ReadVariableOp*^summarize_vars/global_norm/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*U
_classK
IGloc:@loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update/AssignVariableOp_2�
+RMSprop/RMSprop/update_1/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_1_mul_readvariableop_resource*
_output_shapes
:*
dtype02-
+RMSprop/RMSprop/update_1/mul/ReadVariableOp�
RMSprop/RMSprop/update_1/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_1/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_1/mul�
RMSprop/RMSprop/update_1/SquareSquareNgradient_tape/loss/QNetwork/EncodingNetwork/dense/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2!
RMSprop/RMSprop/update_1/Square�
RMSprop/RMSprop/update_1/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_1/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_1/mul_1�
RMSprop/RMSprop/update_1/addAddV2 RMSprop/RMSprop/update_1/mul:z:0"RMSprop/RMSprop/update_1/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_1/add�
)RMSprop/RMSprop/update_1/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_1_mul_readvariableop_resource RMSprop/RMSprop/update_1/add:z:0,^RMSprop/RMSprop/update_1/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_1/AssignVariableOp�
-RMSprop/RMSprop/update_1/mul_2/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_1_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02/
-RMSprop/RMSprop/update_1/mul_2/ReadVariableOp�
RMSprop/RMSprop/update_1/mul_2MulRMSprop/Identity_1:output:05RMSprop/RMSprop/update_1/mul_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_1/mul_2�
RMSprop/RMSprop/update_1/mul_3MulRMSprop/sub:z:0Ngradient_tape/loss/QNetwork/EncodingNetwork/dense/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_1/mul_3�
RMSprop/RMSprop/update_1/add_1AddV2"RMSprop/RMSprop/update_1/mul_2:z:0"RMSprop/RMSprop/update_1/mul_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_1/add_1�
+RMSprop/RMSprop/update_1/AssignVariableOp_1AssignVariableOp6rmsprop_rmsprop_update_1_mul_2_readvariableop_resource"RMSprop/RMSprop/update_1/add_1:z:0.^RMSprop/RMSprop/update_1/mul_2/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_1/AssignVariableOp_1�
0RMSprop/RMSprop/update_1/Square_1/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_1_mul_2_readvariableop_resource,^RMSprop/RMSprop/update_1/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:*
dtype022
0RMSprop/RMSprop/update_1/Square_1/ReadVariableOp�
!RMSprop/RMSprop/update_1/Square_1Square8RMSprop/RMSprop/update_1/Square_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2#
!RMSprop/RMSprop/update_1/Square_1�
'RMSprop/RMSprop/update_1/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_1_mul_readvariableop_resource*^RMSprop/RMSprop/update_1/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:*
dtype02)
'RMSprop/RMSprop/update_1/ReadVariableOp�
RMSprop/RMSprop/update_1/subSub/RMSprop/RMSprop/update_1/ReadVariableOp:value:0%RMSprop/RMSprop/update_1/Square_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_1/sub�
RMSprop/RMSprop/update_1/mul_4MulRMSprop/Identity:output:0Ngradient_tape/loss/QNetwork/EncodingNetwork/dense/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_1/mul_4�
RMSprop/RMSprop/update_1/SqrtSqrt RMSprop/RMSprop/update_1/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_1/Sqrt�
RMSprop/RMSprop/update_1/add_2AddV2!RMSprop/RMSprop/update_1/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_1/add_2�
 RMSprop/RMSprop/update_1/truedivRealDiv"RMSprop/RMSprop/update_1/mul_4:z:0"RMSprop/RMSprop/update_1/add_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2"
 RMSprop/RMSprop/update_1/truediv�
)RMSprop/RMSprop/update_1/ReadVariableOp_1ReadVariableOpCloss_qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)RMSprop/RMSprop/update_1/ReadVariableOp_1�
RMSprop/RMSprop/update_1/sub_1Sub1RMSprop/RMSprop/update_1/ReadVariableOp_1:value:0$RMSprop/RMSprop/update_1/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_1/sub_1�
+RMSprop/RMSprop/update_1/AssignVariableOp_2AssignVariableOpCloss_qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource"RMSprop/RMSprop/update_1/sub_1:z:0*^RMSprop/RMSprop/update_1/ReadVariableOp_1(^Variables/StopGradient_1/ReadVariableOp;^loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp-^summarize_vars/StopGradient_1/ReadVariableOp,^summarize_vars/global_norm_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*V
_classL
JHloc:@loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_1/AssignVariableOp_2�
+RMSprop/RMSprop/update_2/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_2_mul_readvariableop_resource*
_output_shapes

:*
dtype02-
+RMSprop/RMSprop/update_2/mul/ReadVariableOp�
RMSprop/RMSprop/update_2/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_2/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2
RMSprop/RMSprop/update_2/mul�
RMSprop/RMSprop/update_2/SquareSquareFgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2!
RMSprop/RMSprop/update_2/Square�
RMSprop/RMSprop/update_2/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_2/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_2/mul_1�
RMSprop/RMSprop/update_2/addAddV2 RMSprop/RMSprop/update_2/mul:z:0"RMSprop/RMSprop/update_2/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2
RMSprop/RMSprop/update_2/add�
)RMSprop/RMSprop/update_2/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_2_mul_readvariableop_resource RMSprop/RMSprop/update_2/add:z:0,^RMSprop/RMSprop/update_2/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_2/AssignVariableOp�
-RMSprop/RMSprop/update_2/mul_2/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_2_mul_2_readvariableop_resource*
_output_shapes

:*
dtype02/
-RMSprop/RMSprop/update_2/mul_2/ReadVariableOp�
RMSprop/RMSprop/update_2/mul_2MulRMSprop/Identity_1:output:05RMSprop/RMSprop/update_2/mul_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_2/mul_2�
RMSprop/RMSprop/update_2/mul_3MulRMSprop/sub:z:0Fgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_2/mul_3�
RMSprop/RMSprop/update_2/add_1AddV2"RMSprop/RMSprop/update_2/mul_2:z:0"RMSprop/RMSprop/update_2/mul_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_2/add_1�
+RMSprop/RMSprop/update_2/AssignVariableOp_1AssignVariableOp6rmsprop_rmsprop_update_2_mul_2_readvariableop_resource"RMSprop/RMSprop/update_2/add_1:z:0.^RMSprop/RMSprop/update_2/mul_2/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_2/AssignVariableOp_1�
0RMSprop/RMSprop/update_2/Square_1/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_2_mul_2_readvariableop_resource,^RMSprop/RMSprop/update_2/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:*
dtype022
0RMSprop/RMSprop/update_2/Square_1/ReadVariableOp�
!RMSprop/RMSprop/update_2/Square_1Square8RMSprop/RMSprop/update_2/Square_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2#
!RMSprop/RMSprop/update_2/Square_1�
'RMSprop/RMSprop/update_2/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_2_mul_readvariableop_resource*^RMSprop/RMSprop/update_2/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:*
dtype02)
'RMSprop/RMSprop/update_2/ReadVariableOp�
RMSprop/RMSprop/update_2/subSub/RMSprop/RMSprop/update_2/ReadVariableOp:value:0%RMSprop/RMSprop/update_2/Square_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2
RMSprop/RMSprop/update_2/sub�
RMSprop/RMSprop/update_2/mul_4MulRMSprop/Identity:output:0Fgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_2/mul_4�
RMSprop/RMSprop/update_2/SqrtSqrt RMSprop/RMSprop/update_2/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2
RMSprop/RMSprop/update_2/Sqrt�
RMSprop/RMSprop/update_2/add_2AddV2!RMSprop/RMSprop/update_2/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_2/add_2�
 RMSprop/RMSprop/update_2/truedivRealDiv"RMSprop/RMSprop/update_2/mul_4:z:0"RMSprop/RMSprop/update_2/add_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2"
 RMSprop/RMSprop/update_2/truediv�
)RMSprop/RMSprop/update_2/ReadVariableOp_1ReadVariableOpDloss_qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)RMSprop/RMSprop/update_2/ReadVariableOp_1�
RMSprop/RMSprop/update_2/sub_1Sub1RMSprop/RMSprop/update_2/ReadVariableOp_1:value:0$RMSprop/RMSprop/update_2/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_2/sub_1�
+RMSprop/RMSprop/update_2/AssignVariableOp_2AssignVariableOpDloss_qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource"RMSprop/RMSprop/update_2/sub_1:z:0*^RMSprop/RMSprop/update_2/ReadVariableOp_1(^Variables/StopGradient_2/ReadVariableOp<^loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp-^summarize_vars/StopGradient_2/ReadVariableOp,^summarize_vars/global_norm_2/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*W
_classM
KIloc:@loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_2/AssignVariableOp_2�
+RMSprop/RMSprop/update_3/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_3_mul_readvariableop_resource*
_output_shapes
:*
dtype02-
+RMSprop/RMSprop/update_3/mul/ReadVariableOp�
RMSprop/RMSprop/update_3/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_3/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_3/mul�
RMSprop/RMSprop/update_3/SquareSquarePgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2!
RMSprop/RMSprop/update_3/Square�
RMSprop/RMSprop/update_3/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_3/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_3/mul_1�
RMSprop/RMSprop/update_3/addAddV2 RMSprop/RMSprop/update_3/mul:z:0"RMSprop/RMSprop/update_3/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_3/add�
)RMSprop/RMSprop/update_3/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_3_mul_readvariableop_resource RMSprop/RMSprop/update_3/add:z:0,^RMSprop/RMSprop/update_3/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_3/AssignVariableOp�
-RMSprop/RMSprop/update_3/mul_2/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_3_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02/
-RMSprop/RMSprop/update_3/mul_2/ReadVariableOp�
RMSprop/RMSprop/update_3/mul_2MulRMSprop/Identity_1:output:05RMSprop/RMSprop/update_3/mul_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_3/mul_2�
RMSprop/RMSprop/update_3/mul_3MulRMSprop/sub:z:0Pgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_3/mul_3�
RMSprop/RMSprop/update_3/add_1AddV2"RMSprop/RMSprop/update_3/mul_2:z:0"RMSprop/RMSprop/update_3/mul_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_3/add_1�
+RMSprop/RMSprop/update_3/AssignVariableOp_1AssignVariableOp6rmsprop_rmsprop_update_3_mul_2_readvariableop_resource"RMSprop/RMSprop/update_3/add_1:z:0.^RMSprop/RMSprop/update_3/mul_2/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_3/AssignVariableOp_1�
0RMSprop/RMSprop/update_3/Square_1/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_3_mul_2_readvariableop_resource,^RMSprop/RMSprop/update_3/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:*
dtype022
0RMSprop/RMSprop/update_3/Square_1/ReadVariableOp�
!RMSprop/RMSprop/update_3/Square_1Square8RMSprop/RMSprop/update_3/Square_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2#
!RMSprop/RMSprop/update_3/Square_1�
'RMSprop/RMSprop/update_3/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_3_mul_readvariableop_resource*^RMSprop/RMSprop/update_3/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:*
dtype02)
'RMSprop/RMSprop/update_3/ReadVariableOp�
RMSprop/RMSprop/update_3/subSub/RMSprop/RMSprop/update_3/ReadVariableOp:value:0%RMSprop/RMSprop/update_3/Square_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_3/sub�
RMSprop/RMSprop/update_3/mul_4MulRMSprop/Identity:output:0Pgradient_tape/loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_3/mul_4�
RMSprop/RMSprop/update_3/SqrtSqrt RMSprop/RMSprop/update_3/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_3/Sqrt�
RMSprop/RMSprop/update_3/add_2AddV2!RMSprop/RMSprop/update_3/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_3/add_2�
 RMSprop/RMSprop/update_3/truedivRealDiv"RMSprop/RMSprop/update_3/mul_4:z:0"RMSprop/RMSprop/update_3/add_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2"
 RMSprop/RMSprop/update_3/truediv�
)RMSprop/RMSprop/update_3/ReadVariableOp_1ReadVariableOpEloss_qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)RMSprop/RMSprop/update_3/ReadVariableOp_1�
RMSprop/RMSprop/update_3/sub_1Sub1RMSprop/RMSprop/update_3/ReadVariableOp_1:value:0$RMSprop/RMSprop/update_3/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_3/sub_1�
+RMSprop/RMSprop/update_3/AssignVariableOp_2AssignVariableOpEloss_qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource"RMSprop/RMSprop/update_3/sub_1:z:0*^RMSprop/RMSprop/update_3/ReadVariableOp_1(^Variables/StopGradient_3/ReadVariableOp=^loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp-^summarize_vars/StopGradient_3/ReadVariableOp,^summarize_vars/global_norm_3/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*X
_classN
LJloc:@loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_3/AssignVariableOp_2�
+RMSprop/RMSprop/update_4/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_4_mul_readvariableop_resource*
_output_shapes

:*
dtype02-
+RMSprop/RMSprop/update_4/mul/ReadVariableOp�
RMSprop/RMSprop/update_4/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_4/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
RMSprop/RMSprop/update_4/mul�
RMSprop/RMSprop/update_4/SquareSquare6gradient_tape/loss/QNetwork/dense_2/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2!
RMSprop/RMSprop/update_4/Square�
RMSprop/RMSprop/update_4/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_4/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_4/mul_1�
RMSprop/RMSprop/update_4/addAddV2 RMSprop/RMSprop/update_4/mul:z:0"RMSprop/RMSprop/update_4/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
RMSprop/RMSprop/update_4/add�
)RMSprop/RMSprop/update_4/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_4_mul_readvariableop_resource RMSprop/RMSprop/update_4/add:z:0,^RMSprop/RMSprop/update_4/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_4/AssignVariableOp�
-RMSprop/RMSprop/update_4/mul_2/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_4_mul_2_readvariableop_resource*
_output_shapes

:*
dtype02/
-RMSprop/RMSprop/update_4/mul_2/ReadVariableOp�
RMSprop/RMSprop/update_4/mul_2MulRMSprop/Identity_1:output:05RMSprop/RMSprop/update_4/mul_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_4/mul_2�
RMSprop/RMSprop/update_4/mul_3MulRMSprop/sub:z:06gradient_tape/loss/QNetwork/dense_2/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_4/mul_3�
RMSprop/RMSprop/update_4/add_1AddV2"RMSprop/RMSprop/update_4/mul_2:z:0"RMSprop/RMSprop/update_4/mul_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_4/add_1�
+RMSprop/RMSprop/update_4/AssignVariableOp_1AssignVariableOp6rmsprop_rmsprop_update_4_mul_2_readvariableop_resource"RMSprop/RMSprop/update_4/add_1:z:0.^RMSprop/RMSprop/update_4/mul_2/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_4/AssignVariableOp_1�
0RMSprop/RMSprop/update_4/Square_1/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_4_mul_2_readvariableop_resource,^RMSprop/RMSprop/update_4/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:*
dtype022
0RMSprop/RMSprop/update_4/Square_1/ReadVariableOp�
!RMSprop/RMSprop/update_4/Square_1Square8RMSprop/RMSprop/update_4/Square_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2#
!RMSprop/RMSprop/update_4/Square_1�
'RMSprop/RMSprop/update_4/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_4_mul_readvariableop_resource*^RMSprop/RMSprop/update_4/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:*
dtype02)
'RMSprop/RMSprop/update_4/ReadVariableOp�
RMSprop/RMSprop/update_4/subSub/RMSprop/RMSprop/update_4/ReadVariableOp:value:0%RMSprop/RMSprop/update_4/Square_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
RMSprop/RMSprop/update_4/sub�
RMSprop/RMSprop/update_4/mul_4MulRMSprop/Identity:output:06gradient_tape/loss/QNetwork/dense_2/MatMul_1:product:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_4/mul_4�
RMSprop/RMSprop/update_4/SqrtSqrt RMSprop/RMSprop/update_4/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2
RMSprop/RMSprop/update_4/Sqrt�
RMSprop/RMSprop/update_4/add_2AddV2!RMSprop/RMSprop/update_4/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_4/add_2�
 RMSprop/RMSprop/update_4/truedivRealDiv"RMSprop/RMSprop/update_4/mul_4:z:0"RMSprop/RMSprop/update_4/add_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2"
 RMSprop/RMSprop/update_4/truediv�
)RMSprop/RMSprop/update_4/ReadVariableOp_1ReadVariableOp4loss_qnetwork_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)RMSprop/RMSprop/update_4/ReadVariableOp_1�
RMSprop/RMSprop/update_4/sub_1Sub1RMSprop/RMSprop/update_4/ReadVariableOp_1:value:0$RMSprop/RMSprop/update_4/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes

:2 
RMSprop/RMSprop/update_4/sub_1�
+RMSprop/RMSprop/update_4/AssignVariableOp_2AssignVariableOp4loss_qnetwork_dense_2_matmul_readvariableop_resource"RMSprop/RMSprop/update_4/sub_1:z:0*^RMSprop/RMSprop/update_4/ReadVariableOp_1(^Variables/StopGradient_4/ReadVariableOp,^loss/QNetwork/dense_2/MatMul/ReadVariableOp-^summarize_vars/StopGradient_4/ReadVariableOp,^summarize_vars/global_norm_4/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*G
_class=
;9loc:@loss/QNetwork/dense_2/MatMul/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_4/AssignVariableOp_2�
+RMSprop/RMSprop/update_5/mul/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_5_mul_readvariableop_resource*
_output_shapes
:*
dtype02-
+RMSprop/RMSprop/update_5/mul/ReadVariableOp�
RMSprop/RMSprop/update_5/mulMulRMSprop/Identity_1:output:03RMSprop/RMSprop/update_5/mul/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_5/mul�
RMSprop/RMSprop/update_5/SquareSquare@gradient_tape/loss/QNetwork/dense_2/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2!
RMSprop/RMSprop/update_5/Square�
RMSprop/RMSprop/update_5/mul_1MulRMSprop/sub:z:0#RMSprop/RMSprop/update_5/Square:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_5/mul_1�
RMSprop/RMSprop/update_5/addAddV2 RMSprop/RMSprop/update_5/mul:z:0"RMSprop/RMSprop/update_5/mul_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_5/add�
)RMSprop/RMSprop/update_5/AssignVariableOpAssignVariableOp4rmsprop_rmsprop_update_5_mul_readvariableop_resource RMSprop/RMSprop/update_5/add:z:0,^RMSprop/RMSprop/update_5/mul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02+
)RMSprop/RMSprop/update_5/AssignVariableOp�
-RMSprop/RMSprop/update_5/mul_2/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_5_mul_2_readvariableop_resource*
_output_shapes
:*
dtype02/
-RMSprop/RMSprop/update_5/mul_2/ReadVariableOp�
RMSprop/RMSprop/update_5/mul_2MulRMSprop/Identity_1:output:05RMSprop/RMSprop/update_5/mul_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_5/mul_2�
RMSprop/RMSprop/update_5/mul_3MulRMSprop/sub:z:0@gradient_tape/loss/QNetwork/dense_2/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_5/mul_3�
RMSprop/RMSprop/update_5/add_1AddV2"RMSprop/RMSprop/update_5/mul_2:z:0"RMSprop/RMSprop/update_5/mul_3:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_5/add_1�
+RMSprop/RMSprop/update_5/AssignVariableOp_1AssignVariableOp6rmsprop_rmsprop_update_5_mul_2_readvariableop_resource"RMSprop/RMSprop/update_5/add_1:z:0.^RMSprop/RMSprop/update_5/mul_2/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_5/AssignVariableOp_1�
0RMSprop/RMSprop/update_5/Square_1/ReadVariableOpReadVariableOp6rmsprop_rmsprop_update_5_mul_2_readvariableop_resource,^RMSprop/RMSprop/update_5/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:*
dtype022
0RMSprop/RMSprop/update_5/Square_1/ReadVariableOp�
!RMSprop/RMSprop/update_5/Square_1Square8RMSprop/RMSprop/update_5/Square_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2#
!RMSprop/RMSprop/update_5/Square_1�
'RMSprop/RMSprop/update_5/ReadVariableOpReadVariableOp4rmsprop_rmsprop_update_5_mul_readvariableop_resource*^RMSprop/RMSprop/update_5/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:*
dtype02)
'RMSprop/RMSprop/update_5/ReadVariableOp�
RMSprop/RMSprop/update_5/subSub/RMSprop/RMSprop/update_5/ReadVariableOp:value:0%RMSprop/RMSprop/update_5/Square_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_5/sub�
RMSprop/RMSprop/update_5/mul_4MulRMSprop/Identity:output:0@gradient_tape/loss/QNetwork/dense_2/BiasAdd/BiasAddGrad:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_5/mul_4�
RMSprop/RMSprop/update_5/SqrtSqrt RMSprop/RMSprop/update_5/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2
RMSprop/RMSprop/update_5/Sqrt�
RMSprop/RMSprop/update_5/add_2AddV2!RMSprop/RMSprop/update_5/Sqrt:y:0RMSprop/Const:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_5/add_2�
 RMSprop/RMSprop/update_5/truedivRealDiv"RMSprop/RMSprop/update_5/mul_4:z:0"RMSprop/RMSprop/update_5/add_2:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2"
 RMSprop/RMSprop/update_5/truediv�
)RMSprop/RMSprop/update_5/ReadVariableOp_1ReadVariableOp5loss_qnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)RMSprop/RMSprop/update_5/ReadVariableOp_1�
RMSprop/RMSprop/update_5/sub_1Sub1RMSprop/RMSprop/update_5/ReadVariableOp_1:value:0$RMSprop/RMSprop/update_5/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
:2 
RMSprop/RMSprop/update_5/sub_1�
+RMSprop/RMSprop/update_5/AssignVariableOp_2AssignVariableOp5loss_qnetwork_dense_2_biasadd_readvariableop_resource"RMSprop/RMSprop/update_5/sub_1:z:0*^RMSprop/RMSprop/update_5/ReadVariableOp_1(^Variables/StopGradient_5/ReadVariableOp-^loss/QNetwork/dense_2/BiasAdd/ReadVariableOp-^summarize_vars/StopGradient_5/ReadVariableOp,^summarize_vars/global_norm_5/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*H
_class>
<:loc:@loss/QNetwork/dense_2/BiasAdd/ReadVariableOp/resource*
_output_shapes
 *
dtype02-
+RMSprop/RMSprop/update_5/AssignVariableOp_2�
RMSprop/RMSprop/group_depsNoOp*^RMSprop/RMSprop/update/AssignVariableOp_2,^RMSprop/RMSprop/update_1/AssignVariableOp_2,^RMSprop/RMSprop/update_2/AssignVariableOp_2,^RMSprop/RMSprop/update_3/AssignVariableOp_2,^RMSprop/RMSprop/update_4/AssignVariableOp_2,^RMSprop/RMSprop/update_5/AssignVariableOp_2",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
RMSprop/RMSprop/group_deps�
RMSprop/RMSprop/ConstConst^RMSprop/RMSprop/group_deps*
_output_shapes
: *
dtype0	*
value	B	 R2
RMSprop/RMSprop/Const�
#RMSprop/RMSprop/AssignAddVariableOpAssignAddVariableOp,rmsprop_rmsprop_assignaddvariableop_resourceRMSprop/RMSprop/Const:output:0*
_output_shapes
 *
dtype0	2%
#RMSprop/RMSprop/AssignAddVariableOpP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const�
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype02
AssignAddVariableOpS
Cast/xConst*
_output_shapes
: *
dtype0*
value
B :�2
Cast/xU
CastCastCast/x:output:0*

DstT0	*

SrcT0*
_output_shapes
: 2
CastT
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R2	
Const_1�
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceConst_1:output:0*
_output_shapes
 *
dtype0	2
AssignAddVariableOp_1�
FloorMod/ReadVariableOpReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0	2
FloorMod/ReadVariableOpl
FloorModFloorModFloorMod/ReadVariableOp:value:0Cast:y:0*
T0	*
_output_shapes
: 2

FloorModT
Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal�
condIf	Equal:z:0Bloss_qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceJloss_targetqnetwork_encodingnetwork_dense_3_matmul_readvariableop_resourceCloss_qnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceKloss_targetqnetwork_encodingnetwork_dense_3_biasadd_readvariableop_resourceDloss_qnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceJloss_targetqnetwork_encodingnetwork_dense_4_matmul_readvariableop_resourceEloss_qnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceKloss_targetqnetwork_encodingnetwork_dense_4_biasadd_readvariableop_resource4loss_qnetwork_dense_2_matmul_readvariableop_resource:loss_targetqnetwork_dense_5_matmul_readvariableop_resource5loss_qnetwork_dense_2_biasadd_readvariableop_resource;loss_targetqnetwork_dense_5_biasadd_readvariableop_resource	Equal:z:0*^RMSprop/RMSprop/update/AssignVariableOp_2,^RMSprop/RMSprop/update_1/AssignVariableOp_2,^RMSprop/RMSprop/update_2/AssignVariableOp_2,^RMSprop/RMSprop/update_3/AssignVariableOp_2,^RMSprop/RMSprop/update_4/AssignVariableOp_2,^RMSprop/RMSprop/update_5/AssignVariableOp_2C^loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpE^loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOpB^loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpD^loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOpC^loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOpE^loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOpB^loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOpD^loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOp3^loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOp5^loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOp2^loss/TargetQNetwork/dense_5/MatMul/ReadVariableOp4^loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOp*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *(
_read_only_resource_inputs

	*&
else_branchR
cond_false_48585011*
output_shapes
: *%
then_branchR
cond_true_485850102
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identity�"
IdentityIdentityloss/add_1:z:0^AssignAddVariableOp^AssignAddVariableOp_1^CheckNumerics^FloorMod/ReadVariableOp^RMSprop/Cast/ReadVariableOp^RMSprop/Cast_1/ReadVariableOp^RMSprop/Cast_2/ReadVariableOp$^RMSprop/RMSprop/AssignAddVariableOp(^RMSprop/RMSprop/update/AssignVariableOp*^RMSprop/RMSprop/update/AssignVariableOp_1*^RMSprop/RMSprop/update/AssignVariableOp_2&^RMSprop/RMSprop/update/ReadVariableOp(^RMSprop/RMSprop/update/ReadVariableOp_1/^RMSprop/RMSprop/update/Square_1/ReadVariableOp*^RMSprop/RMSprop/update/mul/ReadVariableOp,^RMSprop/RMSprop/update/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_1/AssignVariableOp,^RMSprop/RMSprop/update_1/AssignVariableOp_1,^RMSprop/RMSprop/update_1/AssignVariableOp_2(^RMSprop/RMSprop/update_1/ReadVariableOp*^RMSprop/RMSprop/update_1/ReadVariableOp_11^RMSprop/RMSprop/update_1/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_1/mul/ReadVariableOp.^RMSprop/RMSprop/update_1/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_2/AssignVariableOp,^RMSprop/RMSprop/update_2/AssignVariableOp_1,^RMSprop/RMSprop/update_2/AssignVariableOp_2(^RMSprop/RMSprop/update_2/ReadVariableOp*^RMSprop/RMSprop/update_2/ReadVariableOp_11^RMSprop/RMSprop/update_2/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_2/mul/ReadVariableOp.^RMSprop/RMSprop/update_2/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_3/AssignVariableOp,^RMSprop/RMSprop/update_3/AssignVariableOp_1,^RMSprop/RMSprop/update_3/AssignVariableOp_2(^RMSprop/RMSprop/update_3/ReadVariableOp*^RMSprop/RMSprop/update_3/ReadVariableOp_11^RMSprop/RMSprop/update_3/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_3/mul/ReadVariableOp.^RMSprop/RMSprop/update_3/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_4/AssignVariableOp,^RMSprop/RMSprop/update_4/AssignVariableOp_1,^RMSprop/RMSprop/update_4/AssignVariableOp_2(^RMSprop/RMSprop/update_4/ReadVariableOp*^RMSprop/RMSprop/update_4/ReadVariableOp_11^RMSprop/RMSprop/update_4/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_4/mul/ReadVariableOp.^RMSprop/RMSprop/update_4/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_5/AssignVariableOp,^RMSprop/RMSprop/update_5/AssignVariableOp_1,^RMSprop/RMSprop/update_5/AssignVariableOp_2(^RMSprop/RMSprop/update_5/ReadVariableOp*^RMSprop/RMSprop/update_5/ReadVariableOp_11^RMSprop/RMSprop/update_5/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_5/mul/ReadVariableOp.^RMSprop/RMSprop/update_5/mul_2/ReadVariableOp&^Variables/StopGradient/ReadVariableOp(^Variables/StopGradient_1/ReadVariableOp(^Variables/StopGradient_2/ReadVariableOp(^Variables/StopGradient_3/ReadVariableOp(^Variables/StopGradient_4/ReadVariableOp(^Variables/StopGradient_5/ReadVariableOp^cond;^loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:^loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp=^loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp<^loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp-^loss/QNetwork/dense_2/BiasAdd/ReadVariableOp,^loss/QNetwork/dense_2/MatMul/ReadVariableOpC^loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpE^loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOpB^loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpD^loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOpC^loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOpE^loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOpB^loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOpD^loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOp3^loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOp5^loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOp2^loss/TargetQNetwork/dense_5/MatMul/ReadVariableOp4^loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOp+^summarize_vars/StopGradient/ReadVariableOp-^summarize_vars/StopGradient_1/ReadVariableOp-^summarize_vars/StopGradient_2/ReadVariableOp-^summarize_vars/StopGradient_3/ReadVariableOp-^summarize_vars/StopGradient_4/ReadVariableOp-^summarize_vars/StopGradient_5/ReadVariableOp*^summarize_vars/global_norm/ReadVariableOp,^summarize_vars/global_norm_1/ReadVariableOp,^summarize_vars/global_norm_2/ReadVariableOp,^summarize_vars/global_norm_3/ReadVariableOp,^summarize_vars/global_norm_4/ReadVariableOp,^summarize_vars/global_norm_5/ReadVariableOp*
T0*
_output_shapes
: 2

Identity�"

Identity_1Identityloss/mul_4:z:0^AssignAddVariableOp^AssignAddVariableOp_1^CheckNumerics^FloorMod/ReadVariableOp^RMSprop/Cast/ReadVariableOp^RMSprop/Cast_1/ReadVariableOp^RMSprop/Cast_2/ReadVariableOp$^RMSprop/RMSprop/AssignAddVariableOp(^RMSprop/RMSprop/update/AssignVariableOp*^RMSprop/RMSprop/update/AssignVariableOp_1*^RMSprop/RMSprop/update/AssignVariableOp_2&^RMSprop/RMSprop/update/ReadVariableOp(^RMSprop/RMSprop/update/ReadVariableOp_1/^RMSprop/RMSprop/update/Square_1/ReadVariableOp*^RMSprop/RMSprop/update/mul/ReadVariableOp,^RMSprop/RMSprop/update/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_1/AssignVariableOp,^RMSprop/RMSprop/update_1/AssignVariableOp_1,^RMSprop/RMSprop/update_1/AssignVariableOp_2(^RMSprop/RMSprop/update_1/ReadVariableOp*^RMSprop/RMSprop/update_1/ReadVariableOp_11^RMSprop/RMSprop/update_1/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_1/mul/ReadVariableOp.^RMSprop/RMSprop/update_1/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_2/AssignVariableOp,^RMSprop/RMSprop/update_2/AssignVariableOp_1,^RMSprop/RMSprop/update_2/AssignVariableOp_2(^RMSprop/RMSprop/update_2/ReadVariableOp*^RMSprop/RMSprop/update_2/ReadVariableOp_11^RMSprop/RMSprop/update_2/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_2/mul/ReadVariableOp.^RMSprop/RMSprop/update_2/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_3/AssignVariableOp,^RMSprop/RMSprop/update_3/AssignVariableOp_1,^RMSprop/RMSprop/update_3/AssignVariableOp_2(^RMSprop/RMSprop/update_3/ReadVariableOp*^RMSprop/RMSprop/update_3/ReadVariableOp_11^RMSprop/RMSprop/update_3/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_3/mul/ReadVariableOp.^RMSprop/RMSprop/update_3/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_4/AssignVariableOp,^RMSprop/RMSprop/update_4/AssignVariableOp_1,^RMSprop/RMSprop/update_4/AssignVariableOp_2(^RMSprop/RMSprop/update_4/ReadVariableOp*^RMSprop/RMSprop/update_4/ReadVariableOp_11^RMSprop/RMSprop/update_4/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_4/mul/ReadVariableOp.^RMSprop/RMSprop/update_4/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_5/AssignVariableOp,^RMSprop/RMSprop/update_5/AssignVariableOp_1,^RMSprop/RMSprop/update_5/AssignVariableOp_2(^RMSprop/RMSprop/update_5/ReadVariableOp*^RMSprop/RMSprop/update_5/ReadVariableOp_11^RMSprop/RMSprop/update_5/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_5/mul/ReadVariableOp.^RMSprop/RMSprop/update_5/mul_2/ReadVariableOp&^Variables/StopGradient/ReadVariableOp(^Variables/StopGradient_1/ReadVariableOp(^Variables/StopGradient_2/ReadVariableOp(^Variables/StopGradient_3/ReadVariableOp(^Variables/StopGradient_4/ReadVariableOp(^Variables/StopGradient_5/ReadVariableOp^cond;^loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:^loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp=^loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp<^loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp-^loss/QNetwork/dense_2/BiasAdd/ReadVariableOp,^loss/QNetwork/dense_2/MatMul/ReadVariableOpC^loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpE^loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOpB^loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpD^loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOpC^loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOpE^loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOpB^loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOpD^loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOp3^loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOp5^loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOp2^loss/TargetQNetwork/dense_5/MatMul/ReadVariableOp4^loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOp+^summarize_vars/StopGradient/ReadVariableOp-^summarize_vars/StopGradient_1/ReadVariableOp-^summarize_vars/StopGradient_2/ReadVariableOp-^summarize_vars/StopGradient_3/ReadVariableOp-^summarize_vars/StopGradient_4/ReadVariableOp-^summarize_vars/StopGradient_5/ReadVariableOp*^summarize_vars/global_norm/ReadVariableOp,^summarize_vars/global_norm_1/ReadVariableOp,^summarize_vars/global_norm_2/ReadVariableOp,^summarize_vars/global_norm_3/ReadVariableOp,^summarize_vars/global_norm_4/ReadVariableOp,^summarize_vars/global_norm_5/ReadVariableOp*
T0*
_output_shapes	
:�2

Identity_1�"

Identity_2Identityloss/mul_3:z:0^AssignAddVariableOp^AssignAddVariableOp_1^CheckNumerics^FloorMod/ReadVariableOp^RMSprop/Cast/ReadVariableOp^RMSprop/Cast_1/ReadVariableOp^RMSprop/Cast_2/ReadVariableOp$^RMSprop/RMSprop/AssignAddVariableOp(^RMSprop/RMSprop/update/AssignVariableOp*^RMSprop/RMSprop/update/AssignVariableOp_1*^RMSprop/RMSprop/update/AssignVariableOp_2&^RMSprop/RMSprop/update/ReadVariableOp(^RMSprop/RMSprop/update/ReadVariableOp_1/^RMSprop/RMSprop/update/Square_1/ReadVariableOp*^RMSprop/RMSprop/update/mul/ReadVariableOp,^RMSprop/RMSprop/update/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_1/AssignVariableOp,^RMSprop/RMSprop/update_1/AssignVariableOp_1,^RMSprop/RMSprop/update_1/AssignVariableOp_2(^RMSprop/RMSprop/update_1/ReadVariableOp*^RMSprop/RMSprop/update_1/ReadVariableOp_11^RMSprop/RMSprop/update_1/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_1/mul/ReadVariableOp.^RMSprop/RMSprop/update_1/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_2/AssignVariableOp,^RMSprop/RMSprop/update_2/AssignVariableOp_1,^RMSprop/RMSprop/update_2/AssignVariableOp_2(^RMSprop/RMSprop/update_2/ReadVariableOp*^RMSprop/RMSprop/update_2/ReadVariableOp_11^RMSprop/RMSprop/update_2/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_2/mul/ReadVariableOp.^RMSprop/RMSprop/update_2/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_3/AssignVariableOp,^RMSprop/RMSprop/update_3/AssignVariableOp_1,^RMSprop/RMSprop/update_3/AssignVariableOp_2(^RMSprop/RMSprop/update_3/ReadVariableOp*^RMSprop/RMSprop/update_3/ReadVariableOp_11^RMSprop/RMSprop/update_3/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_3/mul/ReadVariableOp.^RMSprop/RMSprop/update_3/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_4/AssignVariableOp,^RMSprop/RMSprop/update_4/AssignVariableOp_1,^RMSprop/RMSprop/update_4/AssignVariableOp_2(^RMSprop/RMSprop/update_4/ReadVariableOp*^RMSprop/RMSprop/update_4/ReadVariableOp_11^RMSprop/RMSprop/update_4/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_4/mul/ReadVariableOp.^RMSprop/RMSprop/update_4/mul_2/ReadVariableOp*^RMSprop/RMSprop/update_5/AssignVariableOp,^RMSprop/RMSprop/update_5/AssignVariableOp_1,^RMSprop/RMSprop/update_5/AssignVariableOp_2(^RMSprop/RMSprop/update_5/ReadVariableOp*^RMSprop/RMSprop/update_5/ReadVariableOp_11^RMSprop/RMSprop/update_5/Square_1/ReadVariableOp,^RMSprop/RMSprop/update_5/mul/ReadVariableOp.^RMSprop/RMSprop/update_5/mul_2/ReadVariableOp&^Variables/StopGradient/ReadVariableOp(^Variables/StopGradient_1/ReadVariableOp(^Variables/StopGradient_2/ReadVariableOp(^Variables/StopGradient_3/ReadVariableOp(^Variables/StopGradient_4/ReadVariableOp(^Variables/StopGradient_5/ReadVariableOp^cond;^loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:^loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp=^loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp<^loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp-^loss/QNetwork/dense_2/BiasAdd/ReadVariableOp,^loss/QNetwork/dense_2/MatMul/ReadVariableOpC^loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpE^loss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOpB^loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpD^loss/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOpC^loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOpE^loss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOpB^loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOpD^loss/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOp3^loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOp5^loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOp2^loss/TargetQNetwork/dense_5/MatMul/ReadVariableOp4^loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOp+^summarize_vars/StopGradient/ReadVariableOp-^summarize_vars/StopGradient_1/ReadVariableOp-^summarize_vars/StopGradient_2/ReadVariableOp-^summarize_vars/StopGradient_3/ReadVariableOp-^summarize_vars/StopGradient_4/ReadVariableOp-^summarize_vars/StopGradient_5/ReadVariableOp*^summarize_vars/global_norm/ReadVariableOp,^summarize_vars/global_norm_1/ReadVariableOp,^summarize_vars/global_norm_2/ReadVariableOp,^summarize_vars/global_norm_3/ReadVariableOp,^summarize_vars/global_norm_4/ReadVariableOp,^summarize_vars/global_norm_5/ReadVariableOp*
T0*
_output_shapes	
:�2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*�
_input_shapes�
�:	�:�	:	�:	�:	�:	�::::::::::::::::::::::::::::::2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_12
CheckNumericsCheckNumerics22
FloorMod/ReadVariableOpFloorMod/ReadVariableOp2:
RMSprop/Cast/ReadVariableOpRMSprop/Cast/ReadVariableOp2>
RMSprop/Cast_1/ReadVariableOpRMSprop/Cast_1/ReadVariableOp2>
RMSprop/Cast_2/ReadVariableOpRMSprop/Cast_2/ReadVariableOp2J
#RMSprop/RMSprop/AssignAddVariableOp#RMSprop/RMSprop/AssignAddVariableOp2R
'RMSprop/RMSprop/update/AssignVariableOp'RMSprop/RMSprop/update/AssignVariableOp2V
)RMSprop/RMSprop/update/AssignVariableOp_1)RMSprop/RMSprop/update/AssignVariableOp_12V
)RMSprop/RMSprop/update/AssignVariableOp_2)RMSprop/RMSprop/update/AssignVariableOp_22N
%RMSprop/RMSprop/update/ReadVariableOp%RMSprop/RMSprop/update/ReadVariableOp2R
'RMSprop/RMSprop/update/ReadVariableOp_1'RMSprop/RMSprop/update/ReadVariableOp_12`
.RMSprop/RMSprop/update/Square_1/ReadVariableOp.RMSprop/RMSprop/update/Square_1/ReadVariableOp2V
)RMSprop/RMSprop/update/mul/ReadVariableOp)RMSprop/RMSprop/update/mul/ReadVariableOp2Z
+RMSprop/RMSprop/update/mul_2/ReadVariableOp+RMSprop/RMSprop/update/mul_2/ReadVariableOp2V
)RMSprop/RMSprop/update_1/AssignVariableOp)RMSprop/RMSprop/update_1/AssignVariableOp2Z
+RMSprop/RMSprop/update_1/AssignVariableOp_1+RMSprop/RMSprop/update_1/AssignVariableOp_12Z
+RMSprop/RMSprop/update_1/AssignVariableOp_2+RMSprop/RMSprop/update_1/AssignVariableOp_22R
'RMSprop/RMSprop/update_1/ReadVariableOp'RMSprop/RMSprop/update_1/ReadVariableOp2V
)RMSprop/RMSprop/update_1/ReadVariableOp_1)RMSprop/RMSprop/update_1/ReadVariableOp_12d
0RMSprop/RMSprop/update_1/Square_1/ReadVariableOp0RMSprop/RMSprop/update_1/Square_1/ReadVariableOp2Z
+RMSprop/RMSprop/update_1/mul/ReadVariableOp+RMSprop/RMSprop/update_1/mul/ReadVariableOp2^
-RMSprop/RMSprop/update_1/mul_2/ReadVariableOp-RMSprop/RMSprop/update_1/mul_2/ReadVariableOp2V
)RMSprop/RMSprop/update_2/AssignVariableOp)RMSprop/RMSprop/update_2/AssignVariableOp2Z
+RMSprop/RMSprop/update_2/AssignVariableOp_1+RMSprop/RMSprop/update_2/AssignVariableOp_12Z
+RMSprop/RMSprop/update_2/AssignVariableOp_2+RMSprop/RMSprop/update_2/AssignVariableOp_22R
'RMSprop/RMSprop/update_2/ReadVariableOp'RMSprop/RMSprop/update_2/ReadVariableOp2V
)RMSprop/RMSprop/update_2/ReadVariableOp_1)RMSprop/RMSprop/update_2/ReadVariableOp_12d
0RMSprop/RMSprop/update_2/Square_1/ReadVariableOp0RMSprop/RMSprop/update_2/Square_1/ReadVariableOp2Z
+RMSprop/RMSprop/update_2/mul/ReadVariableOp+RMSprop/RMSprop/update_2/mul/ReadVariableOp2^
-RMSprop/RMSprop/update_2/mul_2/ReadVariableOp-RMSprop/RMSprop/update_2/mul_2/ReadVariableOp2V
)RMSprop/RMSprop/update_3/AssignVariableOp)RMSprop/RMSprop/update_3/AssignVariableOp2Z
+RMSprop/RMSprop/update_3/AssignVariableOp_1+RMSprop/RMSprop/update_3/AssignVariableOp_12Z
+RMSprop/RMSprop/update_3/AssignVariableOp_2+RMSprop/RMSprop/update_3/AssignVariableOp_22R
'RMSprop/RMSprop/update_3/ReadVariableOp'RMSprop/RMSprop/update_3/ReadVariableOp2V
)RMSprop/RMSprop/update_3/ReadVariableOp_1)RMSprop/RMSprop/update_3/ReadVariableOp_12d
0RMSprop/RMSprop/update_3/Square_1/ReadVariableOp0RMSprop/RMSprop/update_3/Square_1/ReadVariableOp2Z
+RMSprop/RMSprop/update_3/mul/ReadVariableOp+RMSprop/RMSprop/update_3/mul/ReadVariableOp2^
-RMSprop/RMSprop/update_3/mul_2/ReadVariableOp-RMSprop/RMSprop/update_3/mul_2/ReadVariableOp2V
)RMSprop/RMSprop/update_4/AssignVariableOp)RMSprop/RMSprop/update_4/AssignVariableOp2Z
+RMSprop/RMSprop/update_4/AssignVariableOp_1+RMSprop/RMSprop/update_4/AssignVariableOp_12Z
+RMSprop/RMSprop/update_4/AssignVariableOp_2+RMSprop/RMSprop/update_4/AssignVariableOp_22R
'RMSprop/RMSprop/update_4/ReadVariableOp'RMSprop/RMSprop/update_4/ReadVariableOp2V
)RMSprop/RMSprop/update_4/ReadVariableOp_1)RMSprop/RMSprop/update_4/ReadVariableOp_12d
0RMSprop/RMSprop/update_4/Square_1/ReadVariableOp0RMSprop/RMSprop/update_4/Square_1/ReadVariableOp2Z
+RMSprop/RMSprop/update_4/mul/ReadVariableOp+RMSprop/RMSprop/update_4/mul/ReadVariableOp2^
-RMSprop/RMSprop/update_4/mul_2/ReadVariableOp-RMSprop/RMSprop/update_4/mul_2/ReadVariableOp2V
)RMSprop/RMSprop/update_5/AssignVariableOp)RMSprop/RMSprop/update_5/AssignVariableOp2Z
+RMSprop/RMSprop/update_5/AssignVariableOp_1+RMSprop/RMSprop/update_5/AssignVariableOp_12Z
+RMSprop/RMSprop/update_5/AssignVariableOp_2+RMSprop/RMSprop/update_5/AssignVariableOp_22R
'RMSprop/RMSprop/update_5/ReadVariableOp'RMSprop/RMSprop/update_5/ReadVariableOp2V
)RMSprop/RMSprop/update_5/ReadVariableOp_1)RMSprop/RMSprop/update_5/ReadVariableOp_12d
0RMSprop/RMSprop/update_5/Square_1/ReadVariableOp0RMSprop/RMSprop/update_5/Square_1/ReadVariableOp2Z
+RMSprop/RMSprop/update_5/mul/ReadVariableOp+RMSprop/RMSprop/update_5/mul/ReadVariableOp2^
-RMSprop/RMSprop/update_5/mul_2/ReadVariableOp-RMSprop/RMSprop/update_5/mul_2/ReadVariableOp2N
%Variables/StopGradient/ReadVariableOp%Variables/StopGradient/ReadVariableOp2R
'Variables/StopGradient_1/ReadVariableOp'Variables/StopGradient_1/ReadVariableOp2R
'Variables/StopGradient_2/ReadVariableOp'Variables/StopGradient_2/ReadVariableOp2R
'Variables/StopGradient_3/ReadVariableOp'Variables/StopGradient_3/ReadVariableOp2R
'Variables/StopGradient_4/ReadVariableOp'Variables/StopGradient_4/ReadVariableOp2R
'Variables/StopGradient_5/ReadVariableOp'Variables/StopGradient_5/ReadVariableOp2
condcond2x
:loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:loss/QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2v
9loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp9loss/QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2|
<loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp<loss/QNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2z
;loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp;loss/QNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2\
,loss/QNetwork/dense_2/BiasAdd/ReadVariableOp,loss/QNetwork/dense_2/BiasAdd/ReadVariableOp2Z
+loss/QNetwork/dense_2/MatMul/ReadVariableOp+loss/QNetwork/dense_2/MatMul/ReadVariableOp2�
Bloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOpBloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd/ReadVariableOp2�
Dloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOpDloss/TargetQNetwork/EncodingNetwork/dense_3/BiasAdd_1/ReadVariableOp2�
Aloss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOpAloss/TargetQNetwork/EncodingNetwork/dense_3/MatMul/ReadVariableOp2�
Closs/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOpCloss/TargetQNetwork/EncodingNetwork/dense_3/MatMul_1/ReadVariableOp2�
Bloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOpBloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd/ReadVariableOp2�
Dloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOpDloss/TargetQNetwork/EncodingNetwork/dense_4/BiasAdd_1/ReadVariableOp2�
Aloss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOpAloss/TargetQNetwork/EncodingNetwork/dense_4/MatMul/ReadVariableOp2�
Closs/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOpCloss/TargetQNetwork/EncodingNetwork/dense_4/MatMul_1/ReadVariableOp2h
2loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOp2loss/TargetQNetwork/dense_5/BiasAdd/ReadVariableOp2l
4loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOp4loss/TargetQNetwork/dense_5/BiasAdd_1/ReadVariableOp2f
1loss/TargetQNetwork/dense_5/MatMul/ReadVariableOp1loss/TargetQNetwork/dense_5/MatMul/ReadVariableOp2j
3loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOp3loss/TargetQNetwork/dense_5/MatMul_1/ReadVariableOp2X
*summarize_vars/StopGradient/ReadVariableOp*summarize_vars/StopGradient/ReadVariableOp2\
,summarize_vars/StopGradient_1/ReadVariableOp,summarize_vars/StopGradient_1/ReadVariableOp2\
,summarize_vars/StopGradient_2/ReadVariableOp,summarize_vars/StopGradient_2/ReadVariableOp2\
,summarize_vars/StopGradient_3/ReadVariableOp,summarize_vars/StopGradient_3/ReadVariableOp2\
,summarize_vars/StopGradient_4/ReadVariableOp,summarize_vars/StopGradient_4/ReadVariableOp2\
,summarize_vars/StopGradient_5/ReadVariableOp,summarize_vars/StopGradient_5/ReadVariableOp2V
)summarize_vars/global_norm/ReadVariableOp)summarize_vars/global_norm/ReadVariableOp2Z
+summarize_vars/global_norm_1/ReadVariableOp+summarize_vars/global_norm_1/ReadVariableOp2Z
+summarize_vars/global_norm_2/ReadVariableOp+summarize_vars/global_norm_2/ReadVariableOp2Z
+summarize_vars/global_norm_3/ReadVariableOp+summarize_vars/global_norm_3/ReadVariableOp2Z
+summarize_vars/global_norm_4/ReadVariableOp+summarize_vars/global_norm_4/ReadVariableOp2Z
+summarize_vars/global_norm_5/ReadVariableOp+summarize_vars/global_norm_5/ReadVariableOp:U Q

_output_shapes
:	�
.
_user_specified_nameexperience/step_type:_[
'
_output_shapes
:�	
0
_user_specified_nameexperience/observation:RN

_output_shapes
:	�
+
_user_specified_nameexperience/action:ZV

_output_shapes
:	�
3
_user_specified_nameexperience/next_step_type:RN

_output_shapes
:	�
+
_user_specified_nameexperience/reward:TP

_output_shapes
:	�
-
_user_specified_nameexperience/discount"�J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�

_q_network
_target_q_network

_optimizer
_update_target
_target_greedy_policy
_policy
_collect_policy
_collect_data_context
	_data_context

_train_argspec
_train_step_counter
_as_transition

signatures

�train"
_generic_user_object
�
_encoder
_q_value_layer
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
_encoder
_q_value_layer
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "QNetwork", "name": "TargetQNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
iter
	decay
learning_rate
momentum
rho
)rms�
*rms�
/rms�
0rms�
1rms�
2rms�	)mg�	*mg�	/mg�	0mg�	1mg�	2mg�"
	optimizer
,
_counter"
_generic_user_object
3
 _wrapped_policy"
_generic_user_object
3
!_wrapped_policy"
_generic_user_object
F
"_greedy_policy
#_random_policy"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_dict_wrapper
: 2Variable
1
	_data_context"
_generic_user_object
"
signature_map
�
$_postprocessing_layers
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 30]}}
 "
trackable_list_wrapper
J
/0
01
12
23
)4
*5"
trackable_list_wrapper
J
/0
01
12
23
)4
*5"
trackable_list_wrapper
�
regularization_losses
3metrics
	variables
4layer_metrics
5non_trainable_variables
trainable_variables

6layers
7layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
8_postprocessing_layers
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 30]}}
 "
trackable_list_wrapper
J
C0
D1
E2
F3
=4
>5"
trackable_list_wrapper
J
C0
D1
E2
F3
=4
>5"
trackable_list_wrapper
�
regularization_losses
Gmetrics
	variables
Hlayer_metrics
Inon_trainable_variables
trainable_variables

Jlayers
Klayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
6:4	 2.update_targets/periodic_update_targets/counter
.

_q_network"
_generic_user_object
.

_q_network"
_generic_user_object
3
!_wrapped_policy"
_generic_user_object
"
_generic_user_object
5
L0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
<
/0
01
12
23"
trackable_list_wrapper
�
%regularization_losses
Ometrics
&	variables
Player_metrics
Qnon_trainable_variables
'trainable_variables

Rlayers
Slayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):'2QNetwork/dense_2/kernel
#:!2QNetwork/dense_2/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�
+regularization_losses
Tmetrics
,	variables
Ulayer_metrics
Vnon_trainable_variables
-trainable_variables

Wlayers
Xlayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
7:5	2%QNetwork/EncodingNetwork/dense/kernel
1:/2#QNetwork/EncodingNetwork/dense/bias
9:72'QNetwork/EncodingNetwork/dense_1/kernel
3:12%QNetwork/EncodingNetwork/dense_1/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
Y0
Z1
[2"
trackable_list_wrapper
 "
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
<
C0
D1
E2
F3"
trackable_list_wrapper
�
9regularization_losses
\metrics
:	variables
]layer_metrics
^non_trainable_variables
;trainable_variables

_layers
`layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
/:-2TargetQNetwork/dense_5/kernel
):'2TargetQNetwork/dense_5/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
�
?regularization_losses
ametrics
@	variables
blayer_metrics
cnon_trainable_variables
Atrainable_variables

dlayers
elayer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
?:=	2-TargetQNetwork/EncodingNetwork/dense_3/kernel
9:72+TargetQNetwork/EncodingNetwork/dense_3/bias
?:=2-TargetQNetwork/EncodingNetwork/dense_4/kernel
9:72+TargetQNetwork/EncodingNetwork/dense_4/bias
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
fregularization_losses
g	variables
htrainable_variables
i	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

/kernel
0bias
jregularization_losses
k	variables
ltrainable_variables
m	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 20, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 9]}}
�

1kernel
2bias
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 20]}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rregularization_losses
s	variables
ttrainable_variables
u	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

Ckernel
Dbias
vregularization_losses
w	variables
xtrainable_variables
y	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 20, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 9]}}
�

Ekernel
Fbias
zregularization_losses
{	variables
|trainable_variables
}	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 30, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 20]}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
Y0
Z1
[2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
fregularization_losses
~metrics
g	variables
layer_metrics
�non_trainable_variables
htrainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
�
jregularization_losses
�metrics
k	variables
�layer_metrics
�non_trainable_variables
ltrainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
�
nregularization_losses
�metrics
o	variables
�layer_metrics
�non_trainable_variables
ptrainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rregularization_losses
�metrics
s	variables
�layer_metrics
�non_trainable_variables
ttrainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
�
vregularization_losses
�metrics
w	variables
�layer_metrics
�non_trainable_variables
xtrainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
�
zregularization_losses
�metrics
{	variables
�layer_metrics
�non_trainable_variables
|trainable_variables
�layers
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
3:12#RMSprop/QNetwork/dense_2/kernel/rms
-:+2!RMSprop/QNetwork/dense_2/bias/rms
A:?	21RMSprop/QNetwork/EncodingNetwork/dense/kernel/rms
;:92/RMSprop/QNetwork/EncodingNetwork/dense/bias/rms
C:A23RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/rms
=:;21RMSprop/QNetwork/EncodingNetwork/dense_1/bias/rms
2:02"RMSprop/QNetwork/dense_2/kernel/mg
,:*2 RMSprop/QNetwork/dense_2/bias/mg
@:>	20RMSprop/QNetwork/EncodingNetwork/dense/kernel/mg
::82.RMSprop/QNetwork/EncodingNetwork/dense/bias/mg
B:@22RMSprop/QNetwork/EncodingNetwork/dense_1/kernel/mg
<::20RMSprop/QNetwork/EncodingNetwork/dense_1/bias/mg
�2�
__inference_train_48585055�
���
FullArgSpec,
args$�!
jself
j
experience
	jweights
varargs
 
varkwjkwargs
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference_train_48585055�*/012)*CDEF=>���������������
���
���

Trajectory3
	step_type&�#
experience/step_type	�?
observation0�-
experience/observation�	-
action#� 
experience/action	�
policy_info� =
next_step_type+�(
experience/next_step_type	�-
reward#� 
experience/reward	�1
discount%�"
experience/discount	�

 
� "���
LossInfo
loss�

loss k
extrab�_
DqnLossInfo&
td_loss�
extra/td_loss�(
td_error�
extra/td_error�