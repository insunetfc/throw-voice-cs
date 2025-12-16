# Create a folder to store everything
mkdir -p iam-backup

# List all IAM users
aws iam list-users > iam-backup/users.json

# List all IAM groups
aws iam list-groups > iam-backup/groups.json

# List all IAM roles
aws iam list-roles > iam-backup/roles.json

# List all IAM policies (customer + AWS-managed)
aws iam list-policies --scope All > iam-backup/policies.json

# For each user, dump attached policies + inline policies
for user in $(aws iam list-users --query 'Users[*].UserName' --output text); do
  aws iam list-attached-user-policies --user-name "$user" > "iam-backup/user-${user}-attached-policies.json"
  aws iam list-user-policies --user-name "$user" > "iam-backup/user-${user}-inline-policies.json"
done

# For each role, dump attached + inline policies
for role in $(aws iam list-roles --query 'Roles[*].RoleName' --output text); do
  aws iam list-attached-role-policies --role-name "$role" > "iam-backup/role-${role}-attached-policies.json"
  aws iam list-role-policies --role-name "$role" > "iam-backup/role-${role}-inline-policies.json"
done

# For each group, dump attached + inline policies
for group in $(aws iam list-groups --query 'Groups[*].GroupName' --output text); do
  aws iam list-attached-group-policies --group-name "$group" > "iam-backup/group-${group}-attached-policies.json"
  aws iam list-group-policies --group-name "$group" > "iam-backup/group-${group}-inline-policies.json"
done
