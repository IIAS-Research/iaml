export interface Todo {
  id: number;
  content: string;
}

export interface Meta {
  totalCount: number;
}
export interface ITokenDecode {
  admin: boolean;
  courriel: string;
  exp: number;
  nom_complet: string;
  privileges: {
    habilitations_homepop: string[]
  };
  id_res: string;
}
export interface ILogComponentInfo {
  title: string;
  date: string;
  note?: string;
  cores: {
    type: 'Corrections' | 'Changements';
    list: {
      text: string;
      icon?: 'alert' | 'new';
      priority?: number;
    }[]
  }[]
}
export interface IUser {
  id: number;
  nom_complet: string;
  id_res: string;
  windows: string;
  courriel: string;
  to_s?: string;
  admin: boolean;
  created_at: string;
  updated_at: string;
}